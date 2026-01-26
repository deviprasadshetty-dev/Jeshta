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
        embedding = np.frombuffer(emb_bytes, dtype=np.float32)
        
        return MemoryAtom(
            id=row[0],
            content=row[1],
            embedding=embedding,
            intent_mask=row[3],
            scope_hash=row[4],
            ttl=row[5],
            confidence=row[6],
            refs=json.loads(row[7]),
            created_at=row[8]
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
                    created_at INTEGER NOT NULL
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_scope ON atoms(scope_hash)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON atoms(created_at)")

    def add_atom(self, atom: MemoryAtom):
        emb_bytes = atom.embedding.astype(np.float32).tobytes()
        refs_json = json.dumps(atom.refs)
        
        with self.conn:
            self.conn.execute("""
                INSERT INTO atoms (id, content, embedding, intent_mask, scope_hash, ttl, confidence, refs, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (atom.id, atom.content, emb_bytes, atom.intent_mask, atom.scope_hash, 
                  atom.ttl, atom.confidence, refs_json, atom.created_at))

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
        self.ids = [a.id for a in data]
        if data:
            self.matrix = np.vstack([a.embedding for a in data])
            # Pre-compute norms for cosine similarity
            self.norms = np.linalg.norm(self.matrix, axis=1)
            # Avoid division by zero
            self.norms[self.norms == 0] = 1e-10
        else:
            self.matrix = np.empty((0, 0))
            self.norms = np.array([])
        self.atoms_map = {a.id: a for a in data}

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
        # We already normalized Q, so we divide by stored norms
        cosine_scores = dot_products / self.norms
        
        # Top K Retrieval
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

class DeltaMem:
    def __init__(self, db_path: str = "delta_mem.db"):
        self.db = DeltaDB(db_path)
        
        # Scoring Weights (Fixed per specification)
        self.ALPHA = 0.45  # Cosine Similarity
        self.BETA = 0.25   # Intent Overlap
        self.GAMMA = 0.15  # Time Decay
        self.DELTA = 0.15  # Confidence

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
        return atom.id

    def search(self, query_emb: List[float], intent_mask: int, scope_hash: str = None, top_k: int = 10):
        current_time = int(time.time())
        
        # 1. Retrieval Pruning (Scope + Liveness)
        candidates = self.db.get_all_active_atoms(current_time)
        
        if scope_hash and scope_hash != "*":
            # Strict scope match OR "global" inheritance?
            # User wants "differentiation".
            # Let's support:
            # - scope_hash="proj_A" -> proj_A only (differentiation)
            # - scope_hash="*" -> ALL (searchability across projects)
            # - scope_hash=None -> No filter (same as *)
            candidates = [c for c in candidates if c.scope_hash == scope_hash]

        # 2. Intent Pruning (Hardware accelerated bitwise check)
        # If intent_mask is provided, strictly prioritize or filter atoms that share intent bits
        if intent_mask > 0:
            # Only keep candidates that share AT LEAST ONE intent bit
            # Or we kept them and rely on scoring. 
            # Requirement: "Retrieval prunes memory BEFORE vector search"
            # Strict pruning:
            candidates = [c for c in candidates if (c.intent_mask & intent_mask) > 0]

        if not candidates:
            return []

        # 3. Vector Search (Using Hamming Distance)
        # Quantize Query
        q_vec = np.array(query_emb, dtype=np.float32)
        q_bits = (q_vec > 0).astype(np.uint8)
        q_packed = np.packbits(q_bits)
        
        index = VectorIndex(candidates)
        vec_results = index.search(q_packed, top_k=len(candidates))
        vec_map = {vid: score for vid, score in vec_results}
        
        scored_candidates = []
        for atom in candidates:
            # A. Similarity (Approx Cosine from Hamming)
            sim = vec_map.get(atom.id, 0.0)
            
            # B. Intent Overlap (Normalized Bit Count)
            if intent_mask > 0:
                overlap_bits = (atom.intent_mask & intent_mask).bit_count()
                total_bits = intent_mask.bit_count()
                overlap_score = overlap_bits / total_bits
            else:
                overlap_score = 0.0
                
            # C. Time Decay (Half-life decay)
            # Formula: N(t) = N0 * (1/2)^(t / t_half)
            # Assume t_half = 7 days (604800 seconds) for relevance half-life
            age_seconds = current_time - atom.created_at
            time_score = math.pow(0.5, age_seconds / 604800.0)
            
            # D. Confidence
            conf_score = atom.confidence
            
            # Final Score
            final_score = (
                self.ALPHA * sim +
                self.BETA * overlap_score +
                self.GAMMA * time_score +
                self.DELTA * conf_score
            )
            
            scored_candidates.append((atom, final_score))
            
        # Top K
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]

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
        return self.db.prune_expired(int(time.time()))
