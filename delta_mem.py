import sqlite3
import json
import time
import math
import os
import dataclasses
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

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
    original_dim: int = 0 # Default for backward compatibility or ease

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist(),
            "intent_mask": self.intent_mask,
            "scope_hash": self.scope_hash,
            "ttl": self.ttl,
            "confidence": self.confidence,
            "refs": self.refs,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_row(row: Tuple) -> 'MemoryAtom':
        # row: (id, content, embedding_bytes, intent_mask, scope_hash, ttl, confidence, refs_json, created_at)
        emb_bytes = row[2]
        embedding = np.frombuffer(emb_bytes, dtype=np.uint8)
        
        # Backward compatibility for schema
        original_dim = 0
        if len(row) > 9:
            original_dim = row[9]
            
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
            original_dim=original_dim
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
            
            # Migration: Check for original_dim
            columns = [info[1] for info in self.conn.execute("PRAGMA table_info(atoms)")]
            if "original_dim" not in columns:
                self.conn.execute("ALTER TABLE atoms ADD COLUMN original_dim INTEGER DEFAULT 0")

    def add_atom(self, atom: MemoryAtom):
        emb_bytes = atom.embedding.tobytes()
        refs_json = json.dumps(atom.refs)
        
        with self.conn:
            self.conn.execute("""
                INSERT INTO atoms (id, content, embedding, intent_mask, scope_hash, ttl, confidence, refs, created_at, original_dim)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (atom.id, atom.content, emb_bytes, atom.intent_mask, atom.scope_hash, 
                  atom.ttl, atom.confidence, refs_json, atom.created_at, atom.original_dim))

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
        with self.conn:
            cur = self.conn.execute("DELETE FROM atoms WHERE ttl IS NOT NULL AND ttl < ?", (current_time,))
            return cur.rowcount

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

    def ingest(self, content: str, embedding: List[float], intent_mask: int, 
               scope_hash: str, refs: List[str] = None, ttl: int = None, confidence: float = 1.0):
        
        # Binary Quantization (Float -> Packed Bits)
        # 1. Convert to numpy
        vec = np.array(embedding, dtype=np.float32)
        # 2. Threshold at 0 (Sign bit)
        bits = (vec > 0).astype(np.uint8)
        # 3. Pack bits
        packed = np.packbits(bits)
        
        # Deterministic ID based on content hash and time window for collision avoidance
        atom_id = f"{int(time.time()*1000)}_{abs(hash(content))}"
        if refs is None:
            refs = []
            
        atom = MemoryAtom(
            id=atom_id,
            content=content,
            embedding=packed, # Stored as binary blob
            intent_mask=intent_mask,
            scope_hash=scope_hash,
            ttl=ttl,
            confidence=confidence,
            refs=refs,
            created_at=int(time.time()),
            original_dim=len(embedding)
        )
        self.db.add_atom(atom)
        
        # Update in-memory index
        self.index.add(atom)
        
        return atom.id

    def search(self, query_emb: List[float], intent_mask: int, scope_hash: str = None, top_k: int = 10, use_spreading_activation: bool = True):
        current_time = int(time.time())
        
        # Quantize Query
        q_vec = np.array(query_emb, dtype=np.float32)
        q_bits = (q_vec > 0).astype(np.uint8)
        q_packed = np.packbits(q_bits)
        
        # Search Global `self.index` for everything.
        # This is fast because it's just a matrix multiplication.
        if self.index.matrix.size == 0:
             return []
             
        # Calculate Similarities Globally
        q_norm = np.linalg.norm(q_packed)
        if q_norm == 0: q_norm = 1e-10
        q = q_packed / q_norm
        
        dot_products = np.dot(self.index.matrix, q)
        cosine_scores = dot_products / self.index.norms
        
        # Initial Filtering & Scoring
        initial_candidates = {}
        
        for i, atom_id in enumerate(self.index.ids):
            # Access atom from map (O(1))
            atom = self.index.atoms_map.get(atom_id)
            if not atom: continue

            # Scope
            if scope_hash and scope_hash != "*" and atom.scope_hash != scope_hash:
                continue
            # Intent
            if intent_mask > 0 and (atom.intent_mask & intent_mask) == 0:
                continue
            # TTL
            if atom.ttl is not None and current_time > atom.ttl:
                continue
                
            # If passed filters:
            sim = float(cosine_scores[i])
            
            # Scoring Logic (reused)
            # Intent Overlap
            if intent_mask > 0:
                overlap_bits = (atom.intent_mask & intent_mask).bit_count()
                total_bits = intent_mask.bit_count()
                overlap_score = overlap_bits / total_bits
            else:
                overlap_score = 0.0
                
            # Time Decay
            age_seconds = current_time - atom.created_at
            time_score = math.pow(0.5, age_seconds / 604800.0)
            
            # Confidence
            conf_score = atom.confidence
            
            base_score = (
                self.ALPHA * sim +
                self.BETA * overlap_score +
                self.GAMMA * time_score +
                self.DELTA * conf_score
            )
            
            initial_candidates[atom_id] = {
                "atom": atom,
                "score": base_score,
                "source": "vector"
            }

        # ----------------------------------------------------------------------
        # INTELLIGENT: Spreading Activation
        # ----------------------------------------------------------------------
        final_candidates = initial_candidates.copy()
        
        if use_spreading_activation:
            # We only spread from the top N candidates to avoid exploding the graph
            # Let's say top_k * 2
            sorted_initial = sorted(initial_candidates.values(), key=lambda x: x["score"], reverse=True)[:top_k*2]
            
            for item in sorted_initial:
                atom = item["atom"]
                current_score = item["score"]
                
                # Spread to Refs (Parents) - "Forward Spreading"
                # If I found "Concept B" (Child), "Concept A" (Parent) is likely relevant context.
                for ref_id in atom.refs:
                    parent_atom = self.index.atoms_map.get(ref_id)
                    
                    # Parent must exist and be active
                    if parent_atom and (parent_atom.ttl is None or parent_atom.ttl > current_time):
                        
                        # Calculate activated score
                        activation = current_score * self.SPREADING_DECAY
                        
                        if ref_id in final_candidates:
                            # Boost existing
                            final_candidates[ref_id]["score"] += activation
                            final_candidates[ref_id]["source"] = "vector+spread"
                        else:
                            # Add new activated candidate
                            # We need to calculate its base score (minus vector sim if we want, or just use activation)
                            # For simplicity, we treat activation as the score for purely spread items
                            # But better to calculate its intrinsic score too? 
                            # Let's just add it with the activation score for now.
                            final_candidates[ref_id] = {
                                "atom": parent_atom, 
                                "score": activation,
                                "source": "spread"
                            }

        # Convert to list
        scored_results = []
        for cid, data in final_candidates.items():
            scored_results.append((data["atom"], data["score"]))
            
        # Top K
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

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
