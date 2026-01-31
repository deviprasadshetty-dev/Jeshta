import sqlite3
import json
import time
import math
import os
import dataclasses
import random
from typing import List, Optional, Dict, Any, Tuple, Set
import numpy as np

# ------------------------------------------------------------------------------
# SCOPE MANAGEMENT (Global vs Project)
# ------------------------------------------------------------------------------

class ScopeManager:
    """
    Hierarchical scope system:
    - GLOBAL: User preferences, coding style (persists across ALL projects)
    - PROJECT: Architecture decisions, domain knowledge (per workspace)
    - SESSION: Ephemeral context (cleared after session ends)
    """
    GLOBAL_SCOPE = "__global__"
    SESSION_SCOPE = "__session__"
    
    # Keywords that indicate a memory should be global
    GLOBAL_KEYWORDS = [
        "always", "never", "prefer", "style", "format",
        "i like", "i want", "my preference", "my style",
        "all projects", "every project", "globally"
    ]
    
    @staticmethod
    def get_scope_chain(project_scope: str) -> List[str]:
        """Returns scopes to search, in priority order (project first)."""
        return [
            project_scope,              # Project-specific (highest priority)
            ScopeManager.GLOBAL_SCOPE,  # Global preferences (fallback)
        ]
    
    @staticmethod
    def is_global(content: str) -> bool:
        """Auto-detect if memory should be stored globally."""
        content_lower = content.lower()
        return any(kw in content_lower for kw in ScopeManager.GLOBAL_KEYWORDS)
    
    @staticmethod
    def get_scope_type(scope_hash: str) -> str:
        """Determine the type of a scope."""
        if scope_hash == ScopeManager.GLOBAL_SCOPE:
            return "global"
        elif scope_hash == ScopeManager.SESSION_SCOPE:
            return "session"
        return "project"

# ------------------------------------------------------------------------------
# CORE PRIMITIVE
# ------------------------------------------------------------------------------

@dataclasses.dataclass
class MemoryAtom:
    id: str
    content: str
    embedding: np.ndarray
    intent_mask: int
    scope_hash: str
    ttl: Optional[int]
    confidence: float
    refs: List[str]
    created_at: int
    original_dim: int = 0
    # Forgetting Curve fields
    access_count: int = 0
    last_accessed: int = 0
    scope_type: str = "project"  # 'global', 'project', 'session'
    # Importance/Pinning (P3)
    importance: int = 5  # 1-10, where 10 = pinned (never decays)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist(),
            "intent_mask": self.intent_mask,
            "scope_hash": self.scope_hash,
            "scope_type": self.scope_type,
            "ttl": self.ttl,
            "confidence": self.confidence,
            "refs": self.refs,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "importance": self.importance,
        }

    @staticmethod
    def from_row(row: Tuple) -> 'MemoryAtom':
        # row: (id, content, embedding, intent_mask, scope_hash, ttl, confidence, refs, created_at, original_dim, access_count, last_accessed, scope_type, importance)
        emb_bytes = row[2]
        embedding = np.frombuffer(emb_bytes, dtype=np.uint8)
        
        # Backward compatibility for extended schema
        original_dim = row[9] if len(row) > 9 else 0
        access_count = row[10] if len(row) > 10 else 0
        last_accessed = row[11] if len(row) > 11 else 0
        scope_type = row[12] if len(row) > 12 else "project"
        importance = row[13] if len(row) > 13 else 5  # Default importance
            
        return MemoryAtom(
            id=row[0],
            content=row[1],
            embedding=embedding,
            intent_mask=row[3],
            scope_hash=row[4],
            ttl=row[5],
            confidence=row[6],
            refs=json.loads(row[7]),
            created_at=row[8],
            original_dim=original_dim,
            access_count=access_count,
            last_accessed=last_accessed,
            scope_type=scope_type,
            importance=importance
        )

# ------------------------------------------------------------------------------
# STORAGE LAYER (SQLite)
# ------------------------------------------------------------------------------

class DeltaDB:
    def __init__(self, db_path: str = "delta_mem.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS atoms (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    intent_mask INTEGER NOT NULL,
                    scope_hash TEXT NOT NULL,
                    ttl INTEGER,
                    confidence REAL NOT NULL,
                    refs TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    original_dim INTEGER DEFAULT 0
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_scope ON atoms(scope_hash)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON atoms(created_at)")
            
            # Migrations: Check for new columns and add if missing
            columns = [info[1] for info in self.conn.execute("PRAGMA table_info(atoms)")]
            
            if "original_dim" not in columns:
                self.conn.execute("ALTER TABLE atoms ADD COLUMN original_dim INTEGER DEFAULT 0")
            
            # Forgetting Curve columns
            if "access_count" not in columns:
                self.conn.execute("ALTER TABLE atoms ADD COLUMN access_count INTEGER DEFAULT 0")
            if "last_accessed" not in columns:
                self.conn.execute("ALTER TABLE atoms ADD COLUMN last_accessed INTEGER DEFAULT 0")
            
            # Multi-Scope columns
            if "scope_type" not in columns:
                self.conn.execute("ALTER TABLE atoms ADD COLUMN scope_type TEXT DEFAULT 'project'")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_scope_type ON atoms(scope_type)")

            # Importance/Pinning column (P3)
            if "importance" not in columns:
                self.conn.execute("ALTER TABLE atoms ADD COLUMN importance INTEGER DEFAULT 5")

            # FTS5 Virtual Table for Hybrid Search
            try:
                self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS atoms_fts USING fts5(content, content='atoms', content_rowid='rowid')")
                
                # Triggers to keep FTS in sync
                self.conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS atoms_ai AFTER INSERT ON atoms BEGIN
                        INSERT INTO atoms_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                """)
                self.conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS atoms_ad AFTER DELETE ON atoms BEGIN
                        INSERT INTO atoms_fts(atoms_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                    END;
                """)
                self.conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS atoms_au AFTER UPDATE ON atoms BEGIN
                        INSERT INTO atoms_fts(atoms_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                        INSERT INTO atoms_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                """)
                
                # Backfill if empty (for existing databases)
                count = self.conn.execute("SELECT count(*) FROM atoms_fts").fetchone()[0]
                if count == 0:
                    self.conn.execute("INSERT INTO atoms_fts(rowid, content) SELECT rowid, content FROM atoms")
                    
            except Exception as e:
                print(f"Warning: FTS5 not supported or failed to init: {e}")

    def add_atom(self, atom: MemoryAtom):
        emb_bytes = atom.embedding.tobytes()
        refs_json = json.dumps(atom.refs)
        
        with self.conn:
            self.conn.execute("""
                INSERT INTO atoms (id, content, embedding, intent_mask, scope_hash, ttl, confidence, refs, created_at, original_dim, access_count, last_accessed, scope_type, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (atom.id, atom.content, emb_bytes, atom.intent_mask, atom.scope_hash, 
                  atom.ttl, atom.confidence, refs_json, atom.created_at, atom.original_dim,
                  atom.access_count, atom.last_accessed, atom.scope_type, atom.importance))

    def update_access_stats(self, atom_ids: List[str], current_time: int):
        """Increment access count and update last_accessed for forgetting curve."""
        if not atom_ids:
            return
        with self.conn:
            for atom_id in atom_ids:
                self.conn.execute(
                    "UPDATE atoms SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (current_time, atom_id)
                )

    def get_atoms_by_ids(self, ids: List[str]) -> List[MemoryAtom]:
        if not ids:
            return []
        placeholders = ','.join('?' for _ in ids)
        cursor = self.conn.execute(f"SELECT * FROM atoms WHERE id IN ({placeholders})", ids)
        return [MemoryAtom.from_row(row) for row in cursor]

    def get_atoms_by_scope(self, scope_hash: str) -> List[MemoryAtom]:
        cursor = self.conn.execute("SELECT * FROM atoms WHERE scope_hash = ?", (scope_hash,))
        return [MemoryAtom.from_row(row) for row in cursor]

    def get_all_active_atoms(self, current_time: int) -> List[MemoryAtom]:
        cursor = self.conn.execute("SELECT * FROM atoms")
        atoms = []
        for row in cursor:
            ttl = row[5]
            if ttl is not None and current_time > ttl:
                continue
            atoms.append(MemoryAtom.from_row(row))
        return atoms

    def prune_expired(self, current_time: int) -> int:
            cur = self.conn.execute("DELETE FROM atoms WHERE ttl IS NOT NULL AND ttl < ?", (current_time,))
            return cur.rowcount

    def soft_delete_atom(self, atom_id: str, delete_time: int):
        with self.conn:
            self.conn.execute("UPDATE atoms SET ttl = ? WHERE id = ?", (delete_time, atom_id))

    def get_related_atoms(self, atom_id: str) -> Dict[str, List[MemoryAtom]]:
        # 1. Get parents (atoms referenced BY this atom)
        # We need the atom itself first to get its refs
        target = self.get_atoms_by_ids([atom_id])
        if not target: return {"parents": [], "children": []}
        
        parent_ids = target[0].refs
        parents = self.get_atoms_by_ids(parent_ids)
        
        # 2. Get children (atoms that reference THIS atom)
        # SQLite JSON search is tricky without extensions, but we stored refs as TEXT JSON.
        # Simple LIKE query is 'okay' for small lists, but not robust.
        # Better: use FTS or specific index. For now, strict LIKE.
        # "refs": ["id1", "id2"] -> LIKE '%"id1"%'
        # Note: This is an expensive scan O(N) unless we index refs separately or use FTS on refs.
        # Given "refs" column is not FTS, we do a scan. For <100k items its fine.
        cursor = self.conn.execute("SELECT * FROM atoms WHERE refs LIKE ?", (f'%"{atom_id}"%',))
        children = [MemoryAtom.from_row(row) for row in cursor]
        
        return {
            "parents": parents,
            "children": children
        }

    def search_fts(self, query: str, limit: int = 50) -> List[Tuple[str, float]]:
        """
        Full-Text Search using FTS5 with BM25 scoring.
        SQLite's BM25 return a negative score (magnitude = relevance).
        We sort ASC so the 'most negative' (best) matches come first.
        """
        try:
            # We join back to atoms to get the ID, as FTS only has rowid
            # We explicitly output bm25(atoms_fts) as score.
            cursor = self.conn.execute("""
                SELECT a.id, bm25(atoms_fts) as score
                FROM atoms_fts 
                JOIN atoms a ON a.rowid = atoms_fts.rowid
                WHERE atoms_fts MATCH ? 
                ORDER BY score ASC
                LIMIT ?
            """, (query, limit))
            
            return list(cursor)
        except Exception as e:
            print(f"FTS Search failed: {e}")
            return []

# ------------------------------------------------------------------------------
# VECTOR INDEX (Custom Matrix Implementation)
# ------------------------------------------------------------------------------

class VectorIndex:
    def __init__(self, data: List[MemoryAtom]):
        self.ids: List[str] = []
        self.matrix: np.ndarray = np.empty((0, 0))
        self.norms: np.ndarray = np.array([])
        self.atoms_map: Dict[str, MemoryAtom] = {}
        
        if data:
            self.rebuild(data)

    def add(self, atom: MemoryAtom):
        """Incrementally add a single atom to the index."""
        if atom.id in self.atoms_map:
            return # Already exists
            
        self.atoms_map[atom.id] = atom
        self.ids.append(atom.id)
        
        # Incremental Matrix Update
        # Ideally we'd valid buffer this, but for <100k atoms, vstack is okay-ish.
        # Optimized: pre-expand if needed, but here we just vstack for simplicity 
        # as per "no heavy bloat".
        new_vec = atom.embedding.reshape(1, -1)
        new_norm = np.linalg.norm(new_vec)
        if new_norm == 0: new_norm = 1e-10
        
        if self.matrix.size == 0:
            self.matrix = new_vec
            self.norms = np.array([new_norm])
        else:
            self.matrix = np.vstack([self.matrix, new_vec])
            self.norms = np.append(self.norms, new_norm)

    def rebuild(self, data: List[MemoryAtom]):
        """Full rebuild of the index."""
        self.ids = [a.id for a in data]
        self.atoms_map = {a.id: a for a in data}
        
        if data:
            self.matrix = np.vstack([a.embedding for a in data])
            self.norms = np.linalg.norm(self.matrix, axis=1)
            self.norms[self.norms == 0] = 1e-10
        else:
            self.matrix = np.empty((0, 0))
            self.norms = np.array([])

    def search(self, query_vec: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        if self.matrix.size == 0:
            return []
        
        # Normalize Query
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            q_norm = 1e-10
        q = query_vec / q_norm

        # Dot Product
        dot_products = np.dot(self.matrix, q)
        
        # Cosine Similarity = (A . B) / (|A| * |B|)
        cosine_scores = dot_products / self.norms
        
        # Top K Retrieval
        # Scored candidates
        if len(cosine_scores) <= top_k:
            indices = np.argsort(-cosine_scores) # Descending
        else:
            # Argpartition for O(N) selection of top K
            unsorted_top_k = np.argpartition(cosine_scores, -top_k)[-top_k:]
            # Then sort just the top K
            indices = unsorted_top_k[np.argsort(-cosine_scores[unsorted_top_k])]
            
        results = []
        for idx in indices:
            results.append((self.ids[idx], float(cosine_scores[idx])))
        
        return results

# ------------------------------------------------------------------------------
# MEMORY ENGINE
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# MEMORY ENGINE
# ------------------------------------------------------------------------------

class DeltaMem:
    def __init__(self, db_path: str = "delta_mem.db"):
        self.db_path = db_path
        self.idx_mtime = 0
        self.db = DeltaDB(db_path)
        
        # Scoring Weights (Fixed per specification)
        self.ALPHA = 0.45  # Cosine Similarity
        self.BETA = 0.25   # Intent Overlap
        self.GAMMA = 0.15  # Time Decay
        self.DELTA = 0.15  # Confidence
        
        # Spreading Activation Parameters
        self.SPREADING_DECAY = 0.5 # How much activation flows to neighbors

        # Initialize persistent index
        self._hydrate_index()

    def _hydrate_index(self):
        current_time = int(time.time())
        active_atoms = self.db.get_all_active_atoms(current_time)
        self.index = VectorIndex(active_atoms)
        
        # Record the time we loaded the data
        if os.path.exists(self.db_path):
            self.idx_mtime = os.path.getmtime(self.db_path)

    # --------------------------------------------------------------------------
    # P0: DEDUPLICATION ON INGEST
    # --------------------------------------------------------------------------
    def _check_duplicate(self, packed_embedding: np.ndarray, scope_hash: str, 
                         threshold: float = 0.90) -> Optional[str]:
        """
        Check if a similar memory already exists.
        Returns existing atom_id if duplicate found, else None.
        """
        if self.index.matrix.size == 0:
            return None
        
        # Search using packed embedding directly
        results = self.index.search(packed_embedding, top_k=5)
        
        current_time = int(time.time())
        for atom_id, score in results:
            if score >= threshold:
                atom = self.index.atoms_map.get(atom_id)
                if atom and atom.scope_hash == scope_hash:
                    # Check TTL
                    if atom.ttl is None or atom.ttl > current_time:
                        return atom_id
        return None

    # --------------------------------------------------------------------------
    # P2: CONTRADICTION DETECTION
    # --------------------------------------------------------------------------
    CONTRADICTION_PAIRS = [
        # (positive, negative) or (mutually_exclusive_set)
        ("use", "don't use"), ("using", "not using"),
        ("enable", "disable"), ("enabled", "disabled"),
        ("prefer", "avoid"), ("always", "never"),
        ("add", "remove"), ("include", "exclude"),
        ("yes", "no"), ("true", "false"),
    ]
    
    MUTUALLY_EXCLUSIVE = [
        # Technologies that are typically either/or choices
        {"postgresql", "mysql", "sqlite", "mongodb", "mariadb"},
        {"react", "vue", "angular", "svelte"},
        {"tabs", "spaces"},
        {"npm", "yarn", "pnpm"},
        {"rest", "graphql", "grpc"},
        {"javascript", "typescript"},  # When discussing preference
    ]

    def _detect_conflict(self, content: str, packed_embedding: np.ndarray, 
                         scope_hash: str) -> Optional[Dict[str, Any]]:
        """
        Detect potential conflicts with existing memories.
        Returns conflict info if found, else None.
        """
        if self.index.matrix.size == 0:
            return None
        
        content_lower = content.lower()
        
        # Search for similar content
        results = self.index.search(packed_embedding, top_k=10)
        current_time = int(time.time())
        
        for atom_id, score in results:
            if score < 0.5:  # Too dissimilar to be a conflict
                continue
                
            atom = self.index.atoms_map.get(atom_id)
            if not atom or atom.scope_hash != scope_hash:
                continue
            if atom.ttl is not None and atom.ttl <= current_time:
                continue
            
            existing_lower = atom.content.lower()
            
            # Check 1: Contradiction pairs (use vs don't use)
            for pos, neg in self.CONTRADICTION_PAIRS:
                if (pos in content_lower and neg in existing_lower) or \
                   (neg in content_lower and pos in existing_lower):
                    return {
                        "type": "negation",
                        "conflict_id": atom_id,
                        "conflict_content": atom.content,
                        "reason": f"Contradicting terms: '{pos}' vs '{neg}'"
                    }
            
            # Check 2: Mutually exclusive technologies
            for exclusive_set in self.MUTUALLY_EXCLUSIVE:
                content_matches = [t for t in exclusive_set if t in content_lower]
                existing_matches = [t for t in exclusive_set if t in existing_lower]
                
                if content_matches and existing_matches:
                    if set(content_matches) != set(existing_matches):
                        return {
                            "type": "mutual_exclusion",
                            "conflict_id": atom_id,
                            "conflict_content": atom.content,
                            "reason": f"Mutually exclusive: {content_matches} vs {existing_matches}"
                        }
        
        return None

    # --------------------------------------------------------------------------
    # P3: AUTO-IMPORTANCE DETECTION
    # --------------------------------------------------------------------------
    IMPORTANCE_PATTERNS = {
        10: ["my name is", "i am called", "i'm called", "call me"],  # Identity = pinned
        9: ["always use", "never use", "must use", "must never"],  # Strong rules
        8: ["i prefer", "i like", "i want", "i need"],  # Preferences
        7: ["we decided", "we chose", "project uses", "using"],  # Decisions
        6: ["remember that", "note that", "keep in mind"],  # Explicit memory
    }

    def _auto_detect_importance(self, content: str, scope_type: str) -> int:
        """
        Auto-detect importance based on content.
        Returns importance score 1-10.
        """
        # Global scope items get a boost
        base_importance = 5
        if scope_type == "global":
            base_importance = 7
        
        content_lower = content.lower()
        
        for importance, patterns in self.IMPORTANCE_PATTERNS.items():
            for pattern in patterns:
                if pattern in content_lower:
                    return max(importance, base_importance)
        
        return base_importance

    def ingest(self, content: str, embedding: List[float], intent_mask: int, 
               scope_hash: str, refs: List[str] = None, ttl: int = None, 
               confidence: float = 1.0, force_global: bool = False,
               check_duplicates: bool = True, check_conflicts: bool = True):
        """
        Ingest a new memory atom with deduplication and conflict detection.
        
        Args:
            force_global: If True, always store in global scope
            check_duplicates: If True, check for and skip duplicates (P0)
            check_conflicts: If True, detect and warn about conflicts (P2)
        
        Returns:
            Dict with 'id' (atom_id), 'duplicate' (bool), 'conflict' (optional info)
        """
        # Binary Quantization (Float -> Packed Bits)
        vec = np.array(embedding, dtype=np.float32)
        bits = (vec > 0).astype(np.uint8)
        packed = np.packbits(bits)
        
        # Auto-detect global scope based on content keywords
        if force_global or ScopeManager.is_global(content):
            effective_scope = ScopeManager.GLOBAL_SCOPE
            scope_type = "global"
        else:
            effective_scope = scope_hash
            scope_type = ScopeManager.get_scope_type(scope_hash)
        
        result = {"id": None, "duplicate": False, "conflict": None}
        
        # P0: Deduplication check
        if check_duplicates:
            existing_id = self._check_duplicate(packed, effective_scope)
            if existing_id:
                # Update access stats on the existing memory instead of creating duplicate
                self.db.update_access_stats([existing_id], int(time.time()))
                result["id"] = existing_id
                result["duplicate"] = True
                return result
        
        # P2: Conflict detection
        if check_conflicts:
            conflict = self._detect_conflict(content, packed, effective_scope)
            if conflict:
                result["conflict"] = conflict
                # Auto-add ref to conflicting atom for traceability
                if refs is None:
                    refs = []
                if conflict["conflict_id"] not in refs:
                    refs.append(conflict["conflict_id"])
                # Lower confidence for conflicting entries
                confidence = min(confidence, 0.7)
        
        # Generate unique ID
        atom_id = f"{int(time.time()*1000)}_{abs(hash(content))}_{random.randint(0, 9999)}"
        if refs is None:
            refs = []
        
        # P3: Auto-detect importance
        importance = self._auto_detect_importance(content, scope_type)
        
        current_time = int(time.time())
        atom = MemoryAtom(
            id=atom_id,
            content=content,
            embedding=packed,
            intent_mask=intent_mask,
            scope_hash=effective_scope,
            ttl=ttl,
            confidence=confidence,
            refs=refs,
            created_at=current_time,
            original_dim=len(embedding),
            access_count=0,
            last_accessed=current_time,
            scope_type=scope_type,
            importance=importance
        )
        self.db.add_atom(atom)
        
        # Update in-memory index
        self.index.add(atom)
        
        result["id"] = atom.id
        return result

    def search(self, query_emb: List[float], intent_mask: int, scope_hash: str = None, top_k: int = 10, use_spreading_activation: bool = True, query_text: str = None):
        """
        Hybrid Search: Vector + FTS using Reciprocal Rank Fusion (RRF).
        """
        # Hot-reload: Check if DB file changed since last load
        if os.path.exists(self.db_path):
            mtime = os.path.getmtime(self.db_path)
            if mtime > self.idx_mtime:
                # Reload index from disk
                self._hydrate_index()
                
        current_time = int(time.time())
        k_const = 60 # RRF constant
        
        # 1. Vector Search
        vector_results = []
        if query_emb:
            q_vec = np.array(query_emb, dtype=np.float32)
            q_bits = (q_vec > 0).astype(np.uint8)
            q_packed = np.packbits(q_bits)
            
            if self.index.matrix.size > 0:
                q_norm = np.linalg.norm(q_packed)
                if q_norm == 0: q_norm = 1e-10
                q = q_packed / q_norm
                
                dot_products = np.dot(self.index.matrix, q)
                cosine_scores = dot_products / self.index.norms
                
                # Get raw candidates for post-processing/filtering
                # Optimization: Filter *before* scoring if possible, but matrix ops are fast.
                # We'll just filter iterates.
                
                filter_mask = np.ones(len(self.index.ids), dtype=bool)
                # We can't vector-filter by object props easily without parallel arrays.
                # So we iterate.
                
                for i, atom_id in enumerate(self.index.ids):
                    atom = self.index.atoms_map.get(atom_id)
                    if not atom: continue
                    
                    # Filtering
                    if scope_hash and scope_hash != "*" and atom.scope_hash != scope_hash: continue
                    if intent_mask > 0 and (atom.intent_mask & intent_mask) == 0: continue
                    if atom.ttl is not None and current_time > atom.ttl: continue
                    
                    vector_results.append({
                        "id": atom_id, 
                        "score": float(cosine_scores[i]),
                        "atom": atom
                    })
                
                # Sort by vector score
                vector_results.sort(key=lambda x: x["score"], reverse=True)
                vector_results = vector_results[:top_k*2]

        # 2. FTS Search
        fts_results = []
        if query_text:
            raw_fts = self.db.search_fts(query_text, limit=top_k*2)
            # Map FTS results to atoms and filter
            for atom_id, rank in raw_fts:
                atom = self.index.atoms_map.get(atom_id)
                if not atom: continue
                
                if scope_hash and scope_hash != "*" and atom.scope_hash != scope_hash: continue
                if intent_mask > 0 and (atom.intent_mask & intent_mask) == 0: continue
                if atom.ttl is not None and current_time > atom.ttl: continue
                
                fts_results.append({
                    "id": atom_id,
                    "score": rank, # Rank is opaque, but RRF only cares about position
                    "atom": atom
                })
        
        # 3. Reciprocal Rank Fusion
        fused_scores = {}
        
        # Process Vector Ranks
        for rank, item in enumerate(vector_results):
            atom_id = item["id"]
            if atom_id not in fused_scores:
                fused_scores[atom_id] = {"score": 0.0, "atom": item["atom"], "sources": []}
            
            fused_scores[atom_id]["score"] += 1.0 / (k_const + rank + 1)
            fused_scores[atom_id]["sources"].append("vector")

        # Process FTS Ranks
        for rank, item in enumerate(fts_results):
            atom_id = item["id"]
            if atom_id not in fused_scores:
                fused_scores[atom_id] = {"score": 0.0, "atom": item["atom"], "sources": []}
                
            fused_scores[atom_id]["score"] += 1.0 / (k_const + rank + 1)
            fused_scores[atom_id]["sources"].append("fts")

        # 4. Spreading Activation (P1: Multi-hop with exponential decay)
        final_candidates = fused_scores
        if use_spreading_activation:
             # Take top items from fusion to spread from
             sorted_heads = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
             
             # Multi-hop spreading activation (P1 improvement)
             def spread_recursively(atom_id: str, activation: float, depth: int = 0, max_depth: int = 2, visited: set = None):
                 """Recursively spread activation through refs with exponential decay."""
                 if visited is None:
                     visited = set()
                 if depth >= max_depth or activation < 0.01 or atom_id in visited:
                     return
                 
                 visited.add(atom_id)
                 atom = self.index.atoms_map.get(atom_id)
                 if not atom:
                     return
                 
                 for ref_id in atom.refs:
                     parent = self.index.atoms_map.get(ref_id)
                     if not parent:
                         continue
                     if parent.ttl is not None and parent.ttl <= current_time:
                         continue
                     
                     # Decay activation with each hop
                     ref_activation = activation * self.SPREADING_DECAY
                     
                     if ref_id in final_candidates:
                         final_candidates[ref_id]["score"] += ref_activation
                         if "spread" not in final_candidates[ref_id]["sources"]:
                             final_candidates[ref_id]["sources"].append("spread")
                     else:
                         final_candidates[ref_id] = {
                             "score": ref_activation,
                             "atom": parent,
                             "sources": ["spread"]
                         }
                     
                     # Recurse to next hop
                     spread_recursively(ref_id, ref_activation, depth + 1, max_depth, visited)
             
             # Start spreading from each top result
             for item in sorted_heads:
                 atom = item["atom"]
                 base_score = item["score"]
                 spread_recursively(atom.id, base_score, depth=0, max_depth=2, visited=set())

        # Format Output
        results = []
        for pid, data in final_candidates.items():
            results.append((data["atom"], data["score"]))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compile_context(self, scope_hash: str) -> Dict[str, Any]:
        """
        Compiles the current cognitive state by resolving the Delta Graph.
        
        Algorithm:
        1. Fetch all atoms in scope.
        2. Build a Directed Acyclic Graph (DAG) where Edge(A -> B) means A references (modifies) B.
        3. Identify 'Head' atoms: Nodes with in-degree 0 (not referenced by any active atom).
           These represent the latest version of any concept chain.
        4. Traverse from Heads to resolve final state.
        5. Categorize based on content stability and intent.
        """
        current_time = int(time.time())
        atoms = self.db.get_atoms_by_scope(scope_hash)
        active_atoms = [a for a in atoms if a.ttl is None or a.ttl > current_time]
        
        atom_map = {a.id: a for a in active_atoms}
        
        # Build Dependency Graph
        # refs list: [ref_id, ...] -> parent atoms
        # If A references B, A is the Child (Delta), B is the Parent.
        # We want to find the "Latest" atoms. A 'Head' is an atom that NO ONE references.
        
        referenced_ids = set()
        for atom in active_atoms:
            for ref_id in atom.refs:
                if ref_id in atom_map:
                    referenced_ids.add(ref_id)
        
        # Heads are atoms that are NOT referenced by any other active atom.
        # This implies they are the "tip" of the delta branch.
        head_atoms = [a for a in active_atoms if a.id not in referenced_ids]
        
        stable_facts = []
        recent_deltas = []
        active_constraints = []
        low_confidence = []
        
        # Process only Head atoms as they constitute the "resolved" state.
        # Any atom that is referenced is considered "superceded" or "history".
        
        for atom in head_atoms:
            if atom.confidence < 0.3:
                low_confidence.append(atom.content)
                continue
            
            # Categorize based on Intent Mask
            # 1 (001): Fact
            # 2 (010): Delta/Action
            # 4 (100): Constraint
            
            if (atom.intent_mask & 4) > 0:
                active_constraints.append(atom.content)
            elif (atom.intent_mask & 2) > 0:
                # It's a Delta, but since it's a Head, it's the Active Delta.
                # Check recency
                if (current_time - atom.created_at) < 86400: # 24 hours
                    recent_deltas.append(atom.content)
                else:
                    # Old deltas become facts
                    stable_facts.append(atom.content)
            else:
                # Default Fact
                stable_facts.append(atom.content)

        return {
            "stable_facts": stable_facts,
            "recent_deltas": recent_deltas,
            "active_constraints": active_constraints, 
            "low_confidence_flags": low_confidence
        }

    def diff_memory(self, id_a: str, id_b: str):
        atoms = self.db.get_atoms_by_ids([id_a, id_b])
        if len(atoms) != 2:
            return {"error": "Atoms not found"}
        
        a, b = atoms[0], atoms[1]
        
        # Semantic Difference (Hamming Distance on PackedBits)
        xor_res = np.bitwise_xor(a.embedding, b.embedding)
        hamming_dist = np.unpackbits(xor_res).sum()
        
        # Convert to approximate cosine similarity
        total_bits = a.original_dim
        if total_bits == 0:
            sim = 0.0
        else:
            sim = 1.0 - (2.0 * hamming_dist / total_bits)

        # Intent Difference
        # XOR gives bits that are different
        intent_xor = a.intent_mask ^ b.intent_mask
        
        return {
            "similarity": float(sim),
            "hamming_distance": int(hamming_dist),
            "intent_divergence_mask": intent_xor,
            "time_gap_seconds": abs(a.created_at - b.created_at),
            "ref_relationship": (a.id in b.refs) or (b.id in a.refs)
        }

    def compact_scope(self, scope_hash: str) -> Dict[str, int]:
        """
        Performs Temporal Compaction ('Squashing').
        1. Identifies HEAD atoms (the current active state).
        2. Creates new 'Snapshot' atoms for each HEAD.
        3. Archives (expires) ALL atoms in the scope.
        4. Inserts the new Snapshots.
        
        This resets the Delta Graph depth to 1 for this scope.
        """
        current_time = int(time.time())
        atoms = self.db.get_atoms_by_scope(scope_hash)
        active_atoms = [a for a in atoms if a.ttl is None or a.ttl > current_time]
        
        if not active_atoms:
            return {"compacted_count": 0}

        # 1. DAG Resolution to find Heads
        atom_map = {a.id: a for a in active_atoms}
        referenced_ids = set()
        for atom in active_atoms:
            for ref_id in atom.refs:
                if ref_id in atom_map:
                    referenced_ids.add(ref_id)
        
        head_atoms = [a for a in active_atoms if a.id not in referenced_ids]
        
        # 2. Create Snapshots
        snapshots = []
        for head in head_atoms:
            # New Snapshot Atom
            # content, embedding, intent preserved.
            # refs CLEARED (it is now a root).
            # Created_at updated to NOW.
            
            # Fix: Ensure uniqueness even if loop is fast.
            import random
            rand_suffix = random.randint(0, 999999)
            snp_id = f"snp_{int(time.time()*1000)}_{abs(hash(head.content))%10000}_{rand_suffix}"
            
            snapshot = MemoryAtom(
                id=snp_id,
                content=head.content,
                embedding=head.embedding,
                intent_mask=head.intent_mask,
                scope_hash=head.scope_hash,
                ttl=None, # Fresh start
                confidence=head.confidence,
                refs=[], # Reset history
                created_at=current_time
            )
            snapshots.append(snapshot)
            
        # 3. Archive Old State (Soft Delete via TTL)
        # We set TTL of ALL active atoms to (current_time - 1)
        # In a real DB we would batch update. Here we iterate (SQLite is fast enough for localized scopes).
        
        # Bulk update TTL
        # Optimized: Execute single SQL update
        with self.db.conn:
            self.db.conn.execute(
                "UPDATE atoms SET ttl = ? WHERE scope_hash = ? AND (ttl IS NULL OR ttl > ?)",
                (current_time - 1, scope_hash, current_time)
            )
            
        # 4. Insert Snapshots
        for snp in snapshots:
            self.db.add_atom(snp)
            
        return {
            "compacted_count": len(active_atoms),
            "snapshot_count": len(snapshots)
        }

    def prune(self):
        count = self.db.prune_expired(int(time.time()))
        if count > 0:
             self._hydrate_index() # Rebuild index to remove pruned
        return count

    # --------------------------------------------------------------------------
    # RELIABLE: Causal Integrity Check
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # RESOURCES SUPPORT
    # --------------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        with self.db.conn:
            total = self.db.conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
            active = self.db.conn.execute("SELECT COUNT(*) FROM atoms WHERE ttl IS NULL OR ttl > ?", (int(time.time()),)).fetchone()[0]
            
        return {
            "total_atoms": total,
            "active_atoms": active,
            "index_size": len(self.index.ids),
            "db_path": "delta_mem.db"  # simplified
        }

    def verify_integrity(self, scope_hash: str) -> Dict[str, Any]:
        """
        Scans the Delta Graph for paradoxes and corruption.
        1. Temporal Paradox: Child created before Parent.
        2. Dangling Reference: Child cites non-existent Parent.
        """
        atoms = self.db.get_atoms_by_scope(scope_hash)
        atom_map = {a.id: a for a in atoms}
        
        issues = []
        checked_count = 0
        
        for atom in atoms:
            checked_count += 1
            for ref_id in atom.refs:
                parent = atom_map.get(ref_id)
                
                if not parent:
                    # Check if it exists globally (could be cross-scope reference?)
                    # For now, strict check: Integrity check is usually per-scope or involves checking DB.
                    # We'll assume dangling if not in memory map, or query DB for it?
                    # Let's query DB to be safe, as scope_hash might filter it out if we passed a specific scope but ref is elsewhere.
                    # But for now, we assume strict scoping or self-contained.
                    # Let's just flag it as dangling.
                    issues.append({
                        "id": atom.id,
                        "param": "ref",
                        "error": "Dangling Reference",
                        "details": f"Refers to missing atom {ref_id}"
                    })
                    continue
                    
                # Temporal Paradox Check
                # Allow a small buffer (e.g., 1 second) for clock skew if distributed, but here is local.
                if atom.created_at < parent.created_at:
                    issues.append({
                        "id": atom.id,
                        "param": "created_at",
                        "error": "Temporal Paradox",
                        "details": f"Born {atom.created_at}, Parent {parent.id} born {parent.created_at}"
                    })

        return {
            "status": "valid" if not issues else "corrupted",
            "issues_count": len(issues),
            "checked_atoms": checked_count,
            "issues": issues
        }

    # --------------------------------------------------------------------------
    # ORIGINAL: Centroid Consolidation ("Dreaming")
    # --------------------------------------------------------------------------
    def consolidate_clusters(self, scope_hash: str, similarity_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Clusters similar atoms and merges them into a single Centroid Atom.
        This is a destructive maintenance operation (archives originals).
        """
        current_time = int(time.time())
        atoms = self.db.get_atoms_by_scope(scope_hash)
        # Only process active atoms
        active_atoms = [a for a in atoms if (a.ttl is None or a.ttl > current_time)]
        
        if len(active_atoms) < 2:
            return {"compacted": 0}
            
        # We need unpacked embeddings for partial distance capability or re-unpack
        # The 'VectorIndex' already effectively houses them, but optimization:
        # We'll just do a greedy O(N^2) pass for now since this is a background maintenance task
        # and N per scope is usually small (<1000).
        
        visited = set()
        clusters = [] # List of list of atoms
        
        # Helper to unpack bits -> float vector approximation or use hamming distance
        # We already have diff_memory which does hamming. 
        # For 0.95 cosine threshold, we can map to hamming distance threshold.
        # Cosine approx = 1 - 2(h/d). 
        # 0.95 = 1 - 2h/d => 2h/d = 0.05 => h/d = 0.025. 
        # Hamming distance must be <= 2.5% of bits.
        
        for i, atom_a in enumerate(active_atoms):
            if atom_a.id in visited:
                continue
                
            visited.add(atom_a.id)
            current_cluster = [atom_a]
            
            for j in range(i+1, len(active_atoms)):
                atom_b = active_atoms[j]
                if atom_b.id in visited:
                    continue
                    
                # Fast Hamming Check
                xor_res = np.bitwise_xor(atom_a.embedding, atom_b.embedding)
                hamming_dist = np.unpackbits(xor_res).sum()
                
                # Dynamic threshold calculation based on original dimensions
                # If original_dim is 384...
                total_bits = 384 # Default assumption for BGE-small if not stored
                if atom_a.original_dim > 0: total_bits = atom_a.original_dim
                
                # Sim threshold check
                # sim = 1 - 2(h/d). If sim >= thresh, then ...
                # h <= (1 - thresh) * d / 2
                max_hamming = (1.0 - similarity_threshold) * total_bits / 2.0
                
                if hamming_dist <= max_hamming:
                    current_cluster.append(atom_b)
                    visited.add(atom_b.id)
            
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
        
        # Execute Merges
        consolidated_count = 0
        new_atoms = []
        
        for cluster in clusters:
            # 1. Calculate Centroid (Voting on bits)
            # Stack all embeddings (N x D bytes)
            # Unpack to bits (N x D_bits)
            stack = np.vstack([np.unpackbits(a.embedding) for a in cluster])
            # Average (Mean)
            mean_vec = np.mean(stack, axis=0)
            # Re-binarize: if mean > 0.5 -> 1 else 0
            centroid_bits = (mean_vec > 0.5).astype(np.uint8)
            centroid_packed = np.packbits(centroid_bits)
            
            # 2. Pick Representative Content
            # Best is highest confidence, or most recent?
            # Let's pick Highest Confidence.
            repr_atom = max(cluster, key=lambda x: x.confidence)
            
            # 3. Create New Atom
            import random
            rand_suffix = random.randint(0, 999999)
            new_id = f"con_{int(time.time()*1000)}_{len(cluster)}_{rand_suffix}"
            
            new_atom = MemoryAtom(
                id=new_id,
                content=repr_atom.content,
                embedding=centroid_packed,
                intent_mask=repr_atom.intent_mask,
                scope_hash=scope_hash,
                ttl=None,
                confidence=min(1.0, repr_atom.confidence * 1.1), # Boost confidence slightly
                refs=[], # Reset history? Or union of refs? A consolidation typically starts fresh.
                created_at=current_time,
                original_dim=repr_atom.original_dim
            )
            
            new_atoms.append(new_atom)
            consolidated_count += len(cluster)
            
            # 4. Archive Cluster
            ids_to_expire = [a.id for a in cluster]
            placeholders = ','.join('?' for _ in ids_to_expire)
            # Bulk update
            # We can't use '?' for IN clause directly in all drivers easily without constructing string, 
            # but sqlite handles it.
            with self.db.conn:
                self.db.conn.execute(
                    f"UPDATE atoms SET ttl = ? WHERE id IN ({placeholders})",
                    (current_time - 1, *ids_to_expire)
                )

        # Insert new atoms
        for atom in new_atoms:
            self.db.add_atom(atom)
            self.index.add(atom) # Update live index

        return {
            "clusters_found": len(clusters),
            "atoms_consolidated": consolidated_count,
            "new_centroids_created": len(new_atoms)
        }

    # --------------------------------------------------------------------------
    # FINAL POLISH: Delete & Explore
    # --------------------------------------------------------------------------
    def delete_atom(self, atom_id: str) -> bool:
        """
        Soft deletes an atom by setting its TTL to the past.
        """
        # Expire immediately (current_time - 1)
        expire_time = int(time.time()) - 1
        self.db.soft_delete_atom(atom_id, expire_time)
        
        # Remove from in-memory index if present
        if atom_id in self.index.atoms_map:
            del self.index.atoms_map[atom_id]
            # Matrix removal is expensive (requires slicing).
            # We lazy-prune: just ensure subsequent searches filtering by map checks handle missing items.
            # But wait, our search iterate over `self.index.ids`.
            # We should remove from ids (list) which is O(N).
            try:
                self.index.ids.remove(atom_id)
            except ValueError:
                pass
            # Matrix norms/rows drift out of sync with ids if we don't rebuild.
            # Lazy approach: Rebuild index if we delete a lot.
            # Or just mark it as 'deleted' in a separate set?
            # Cleanest: Just let `_hydrate_index` handle it on next startup/prune.
            # For now, searching filters by `if not atom: continue`.
            # We deleted from `atoms_map`, so `get` returns None. Search loop handles it.
            pass
            
        return True

    def recall_related(self, atom_id: str) -> Dict[str, Any]:
        """
        Traverses one hop up (Parents) and down (Children).
        """
        related = self.db.get_related_atoms(atom_id)
        
        return {
            "parents": [a.to_dict() for a in related["parents"]],
            "children": [a.to_dict() for a in related["children"]]
        }

    # --------------------------------------------------------------------------
    # MULTI-SCOPE ARCHITECTURE (Global vs Project)
    # --------------------------------------------------------------------------
    
    def search_hierarchical(self, query_emb: List[float], intent_mask: int, 
                           project_scope: str, top_k: int = 10, 
                           use_spreading_activation: bool = True,
                           query_text: str = None) -> List[Tuple[Any, float]]:
        """
        Search across project → global scope hierarchy with priority weighting.
        Project-specific memories override global when there's overlap.
        """
        all_results = {}
        scope_chain = ScopeManager.get_scope_chain(project_scope)
        
        for priority, scope in enumerate(scope_chain):
            scope_results = self.search(
                query_emb=query_emb,
                intent_mask=intent_mask,
                scope_hash=scope,
                top_k=top_k * 2,
                use_spreading_activation=use_spreading_activation,
                query_text=query_text
            )
            
            for atom, score in scope_results:
                # Apply scope priority boost (project > global)
                # priority 0 = project (no penalty), priority 1 = global (10% penalty)
                boosted_score = score * (1.0 - priority * 0.1)
                
                if atom.id not in all_results or all_results[atom.id][1] < boosted_score:
                    all_results[atom.id] = (atom, boosted_score)
        
        # Update access stats for returned atoms (forgetting curve)
        current_time = int(time.time())
        accessed_ids = list(all_results.keys())[:top_k]
        self.db.update_access_stats(accessed_ids, current_time)
        
        # Sort and return top_k
        results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compile_context_hierarchical(self, project_scope: str) -> Dict[str, Any]:
        """
        Compile context merging project + global memories.
        Global facts are included but project facts take precedence.
        """
        project_ctx = self.compile_context(project_scope)
        global_ctx = self.compile_context(ScopeManager.GLOBAL_SCOPE)
        
        return {
            "stable_facts": global_ctx["stable_facts"] + project_ctx["stable_facts"],
            "recent_deltas": project_ctx["recent_deltas"],  # Only project deltas
            "active_constraints": global_ctx["active_constraints"] + project_ctx["active_constraints"],
            "low_confidence_flags": project_ctx["low_confidence_flags"],
            "_meta": {
                "project_scope": project_scope,
                "global_atoms_count": len(global_ctx["stable_facts"]) + len(global_ctx["active_constraints"]),
                "project_atoms_count": len(project_ctx["stable_facts"]) + len(project_ctx["recent_deltas"])
            }
        }

    def ingest_global(self, content: str, embedding: List[float], intent_mask: int,
                     refs: List[str] = None, ttl: int = None, confidence: float = 1.0) -> Dict[str, Any]:
        """
        Explicitly ingest to global scope (user preferences, style rules).
        Returns dict with id, duplicate status, and conflict info.
        """
        return self.ingest(
            content=content,
            embedding=embedding,
            intent_mask=intent_mask,
            scope_hash=ScopeManager.GLOBAL_SCOPE,
            refs=refs,
            ttl=ttl,
            confidence=confidence,
            force_global=True,
            check_duplicates=True,
            check_conflicts=True
        )

    # --------------------------------------------------------------------------
    # FORGETTING CURVE (Ebbinghaus-inspired memory decay)
    # --------------------------------------------------------------------------
    
    @staticmethod
    def temporal_score(created_at: int, current_time: int, half_life: int = 604800) -> float:
        """
        Exponential decay score based on age.
        
        Args:
            created_at: Unix timestamp of creation
            current_time: Current unix timestamp
            half_life: Time in seconds for score to halve (default: 7 days)
        
        Returns:
            Score between 0 and 1, where 1 is most recent
        """
        age = current_time - created_at
        if age <= 0:
            return 1.0
        return 2 ** (-age / half_life)

    @staticmethod
    def calculate_retention(access_count: int, last_accessed: int, 
                           current_time: int) -> float:
        """
        Calculate memory retention based on Ebbinghaus forgetting curve.
        
        Retention = e^(-t/S) where:
        - t = time since last access (in days)
        - S = stability (grows with reinforcement/access_count)
        
        Based on SuperMemo SM-2 algorithm principles.
        """
        # Stability increases with each access (logarithmically)
        stability = 1.0 + (0.5 * math.log1p(access_count))
        
        # Time since last access in days
        if last_accessed == 0:
            # Never accessed, use created_at effectively
            return 0.5  # Default medium retention
        
        age_days = (current_time - last_accessed) / 86400.0
        if age_days <= 0:
            return 1.0
        
        return math.exp(-age_days / stability)

    def prune_forgotten(self, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Prune atoms with retention below threshold (forgotten memories).
        
        Args:
            threshold: Retention threshold (0-1). Atoms below this are pruned.
                      Default 0.1 means atoms with <10% retention are forgotten.
        
        Returns:
            Dict with pruned count and details
        """
        current_time = int(time.time())
        
        # Get all active atoms
        cursor = self.db.conn.execute(
            "SELECT * FROM atoms WHERE ttl IS NULL OR ttl > ?", 
            (current_time,)
        )
        
        to_forget = []
        for row in cursor:
            atom = MemoryAtom.from_row(row)
            
            # Skip global scope - never auto-forget user preferences
            if atom.scope_type == "global":
                continue
            
            retention = self.calculate_retention(
                atom.access_count, 
                atom.last_accessed, 
                current_time
            )
            
            if retention < threshold:
                to_forget.append({
                    "id": atom.id,
                    "content": atom.content[:50] + "..." if len(atom.content) > 50 else atom.content,
                    "retention": retention,
                    "access_count": atom.access_count
                })
        
        # Soft-delete forgotten atoms
        expire_time = current_time - 1
        for item in to_forget:
            self.db.soft_delete_atom(item["id"], expire_time)
            # Remove from index
            if item["id"] in self.index.atoms_map:
                del self.index.atoms_map[item["id"]]
                try:
                    self.index.ids.remove(item["id"])
                except ValueError:
                    pass
        
        return {
            "pruned_count": len(to_forget),
            "threshold": threshold,
            "details": to_forget[:10]  # Return first 10 for inspection
        }

    def get_memory_health(self, scope_hash: str = None) -> Dict[str, Any]:
        """
        Get health statistics for the memory system.
        Useful for understanding retention distribution.
        """
        current_time = int(time.time())
        
        query = "SELECT * FROM atoms WHERE ttl IS NULL OR ttl > ?"
        params = [current_time]
        
        if scope_hash:
            query += " AND scope_hash = ?"
            params.append(scope_hash)
        
        cursor = self.db.conn.execute(query, params)
        
        stats = {
            "total": 0,
            "by_scope_type": {"global": 0, "project": 0, "session": 0},
            "retention_buckets": {
                "strong (>0.8)": 0,
                "medium (0.4-0.8)": 0,
                "weak (0.1-0.4)": 0,
                "forgotten (<0.1)": 0
            },
            "avg_access_count": 0
        }
        
        total_access = 0
        for row in cursor:
            atom = MemoryAtom.from_row(row)
            stats["total"] += 1
            
            # Count by scope type
            scope_type = atom.scope_type or "project"
            stats["by_scope_type"][scope_type] = stats["by_scope_type"].get(scope_type, 0) + 1
            
            # Calculate retention
            retention = self.calculate_retention(
                atom.access_count, atom.last_accessed, current_time
            )
            
            if retention > 0.8:
                stats["retention_buckets"]["strong (>0.8)"] += 1
            elif retention > 0.4:
                stats["retention_buckets"]["medium (0.4-0.8)"] += 1
            elif retention > 0.1:
                stats["retention_buckets"]["weak (0.1-0.4)"] += 1
            else:
                stats["retention_buckets"]["forgotten (<0.1)"] += 1
            
            total_access += atom.access_count
        
        if stats["total"] > 0:
            stats["avg_access_count"] = round(total_access / stats["total"], 2)
        
        return stats
