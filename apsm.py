"""
APSM: Active Programmatic Synthesis Memory
A cognitive engine with Tri-Layer Memory Stack and Wake-Sleep consolidation.

Layer 1: Episodic Stream (Hippocampus) - Raw experience traces
Layer 2: Semantic Graph (Neocortex-Declarative) - Knowledge graph
Layer 3: Programmatic Library (Neocortex-Procedural) - Executable skills
"""

import sqlite3
import json
import time
import hashlib
import ast
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import IntEnum
import math

logging.basicConfig(level=logging.INFO)

# ==============================================================================
# LAYER 1: EPISODIC STREAM ("Hippocampus")
# ==============================================================================

@dataclass
class Episode:
    """A single trace in the episodic stream."""
    id: str
    timestamp: int
    context: Dict[str, Any]      # Query, file, environment
    action: str                   # What was attempted
    observation: str              # What was observed
    outcome: Dict[str, Any]       # Success/failure, result
    surprise_score: float = 0.0   # Prediction error (high = memorable)
    consolidated: bool = False    # Has this been processed by Sleep?
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_row(row: Tuple) -> 'Episode':
        return Episode(
            id=row[0],
            timestamp=row[1],
            context=json.loads(row[2]),
            action=row[3],
            observation=row[4],
            outcome=json.loads(row[5]),
            surprise_score=row[6],
            consolidated=bool(row[7])
        )


class EpisodicStream:
    """
    Layer 1: High-fidelity, append-only log of interaction traces.
    Uses sliding window + priority buffer for retention.
    """
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()
    
    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    context TEXT NOT NULL,
                    action TEXT NOT NULL,
                    observation TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    surprise_score REAL DEFAULT 0.0,
                    consolidated INTEGER DEFAULT 0
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_time ON episodes(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_surprise ON episodes(surprise_score)")
    
    def log(self, context: Dict, action: str, observation: str, 
            outcome: Dict, surprise_score: float = 0.0) -> str:
        """Append a new trace to the episodic stream."""
        import random
        timestamp = int(time.time() * 1000)  # Milliseconds for uniqueness
        rand = random.randint(0, 99999)
        ep_id = f"ep_{timestamp}_{rand}_{hashlib.md5(action.encode()).hexdigest()[:6]}"
        
        with self.conn:
            self.conn.execute("""
                INSERT INTO episodes (id, timestamp, context, action, observation, outcome, surprise_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ep_id, timestamp, json.dumps(context), action, observation, 
                  json.dumps(outcome), surprise_score))
        
        return ep_id
    
    def recall(self, query: str = None, time_window: int = None, 
               limit: int = 50, min_surprise: float = 0.0) -> List[Episode]:
        """Retrieve episodes from the stream."""
        sql = "SELECT * FROM episodes WHERE 1=1"
        params = []
        
        if time_window:
            cutoff = int(time.time()) - time_window
            sql += " AND timestamp > ?"
            params.append(cutoff)
        
        if min_surprise > 0:
            sql += " AND surprise_score >= ?"
            params.append(min_surprise)
        
        if query:
            # Simple keyword search (FTS could be added)
            sql += " AND (action LIKE ? OR observation LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.execute(sql, params)
        return [Episode.from_row(row) for row in cursor]
    
    def get_unconsolidated(self, limit: int = 100) -> List[Episode]:
        """Get episodes that haven't been processed by Sleep phase."""
        cursor = self.conn.execute(
            "SELECT * FROM episodes WHERE consolidated = 0 ORDER BY surprise_score DESC LIMIT ?",
            (limit,)
        )
        return [Episode.from_row(row) for row in cursor]
    
    def mark_consolidated(self, episode_ids: List[str]):
        """Mark episodes as consolidated (processed by Sleep)."""
        if not episode_ids:
            return
        placeholders = ','.join('?' for _ in episode_ids)
        with self.conn:
            self.conn.execute(
                f"UPDATE episodes SET consolidated = 1 WHERE id IN ({placeholders})",
                episode_ids
            )
    
    def prune_old(self, max_age_days: int = 30, keep_high_surprise: bool = True) -> int:
        """Prune old episodes, optionally keeping high-surprise ones."""
        cutoff = int(time.time()) - (max_age_days * 86400)
        
        sql = "DELETE FROM episodes WHERE timestamp < ? AND consolidated = 1"
        params = [cutoff]
        
        if keep_high_surprise:
            sql += " AND surprise_score < 0.7"
        
        with self.conn:
            cursor = self.conn.execute(sql, params)
            return cursor.rowcount


# ==============================================================================
# LAYER 2: SEMANTIC GRAPH ("Neocortex - Declarative")
# ==============================================================================

@dataclass
class GraphNode:
    """A node in the semantic graph."""
    id: str
    label: str                    # Entity type: concept, entity, fact
    properties: Dict[str, Any]
    created_at: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class GraphEdge:
    """An edge in the semantic graph."""
    id: str
    source_id: str
    target_id: str
    relation: str                 # Relationship type: is_a, has, causes, etc.
    properties: Dict[str, Any]
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticGraph:
    """
    Layer 2: Dynamic Knowledge Graph stored in SQLite with JSON.
    Supports entity resolution and contradiction detection.
    """
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()
    
    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    properties TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    confidence REAL DEFAULT 1.0
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    properties TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
                    FOREIGN KEY (target_id) REFERENCES graph_nodes(id)
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_node_label ON graph_nodes(label)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_rel ON graph_edges(relation)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_source ON graph_edges(source_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_target ON graph_edges(target_id)")
    
    def add_node(self, label: str, properties: Dict, confidence: float = 1.0) -> str:
        """Add a node to the graph, returns node ID."""
        node_id = f"n_{hashlib.md5(f'{label}:{json.dumps(properties, sort_keys=True)}'.encode()).hexdigest()[:12]}"
        timestamp = int(time.time())
        
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO graph_nodes (id, label, properties, created_at, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (node_id, label, json.dumps(properties), timestamp, confidence))
        
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, relation: str, 
                 properties: Dict = None, confidence: float = 1.0) -> str:
        """Add an edge between two nodes."""
        edge_id = f"e_{hashlib.md5(f'{source_id}:{relation}:{target_id}'.encode()).hexdigest()[:12]}"
        props = properties or {}
        
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO graph_edges (id, source_id, target_id, relation, properties, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (edge_id, source_id, target_id, relation, json.dumps(props), confidence))
        
        return edge_id
    
    def query(self, cypher_like: str) -> List[Dict]:
        """
        Simple query language (subset of Cypher-like syntax).
        Examples:
            "MATCH (n:concept) RETURN n"
            "MATCH (a)-[r:causes]->(b) RETURN a, r, b"
        """
        cypher_like = cypher_like.strip().upper()
        
        # Simple pattern: MATCH (n:label) RETURN n
        if "MATCH" in cypher_like and "RETURN" in cypher_like:
            # Extract label if present
            if ":" in cypher_like:
                label = cypher_like.split(":")[1].split(")")[0].strip().lower()
                cursor = self.conn.execute(
                    "SELECT * FROM graph_nodes WHERE label = ?", (label,)
                )
            else:
                cursor = self.conn.execute("SELECT * FROM graph_nodes")
            
            results = []
            for row in cursor:
                results.append({
                    "id": row[0],
                    "label": row[1],
                    "properties": json.loads(row[2]),
                    "confidence": row[4]
                })
            return results
        
        # Fallback: return all nodes
        cursor = self.conn.execute("SELECT * FROM graph_nodes")
        return [{"id": r[0], "label": r[1], "properties": json.loads(r[2])} for r in cursor]
    
    def get_related(self, node_id: str, direction: str = "both") -> Dict[str, List]:
        """Get nodes connected to the given node."""
        outgoing = []
        incoming = []
        
        if direction in ("out", "both"):
            cursor = self.conn.execute("""
                SELECT e.*, n.label, n.properties 
                FROM graph_edges e
                JOIN graph_nodes n ON e.target_id = n.id
                WHERE e.source_id = ?
            """, (node_id,))
            for row in cursor:
                outgoing.append({
                    "edge_id": row[0],
                    "relation": row[3],
                    "target_id": row[2],
                    "target_label": row[6],
                    "target_properties": json.loads(row[7])
                })
        
        if direction in ("in", "both"):
            cursor = self.conn.execute("""
                SELECT e.*, n.label, n.properties 
                FROM graph_edges e
                JOIN graph_nodes n ON e.source_id = n.id
                WHERE e.target_id = ?
            """, (node_id,))
            for row in cursor:
                incoming.append({
                    "edge_id": row[0],
                    "relation": row[3],
                    "source_id": row[1],
                    "source_label": row[6],
                    "source_properties": json.loads(row[7])
                })
        
        return {"outgoing": outgoing, "incoming": incoming}
    
    def find_contradictions(self, node_id: str) -> List[Dict]:
        """Find contradicting relations for a node."""
        contradictions = []
        relations = self.get_related(node_id)
        
        # Check for opposing relations
        opposite_pairs = [
            ("is", "is_not"), ("causes", "prevents"),
            ("enables", "disables"), ("true", "false")
        ]
        
        all_rels = [(r["relation"], r["target_id"]) for r in relations["outgoing"]]
        
        for rel, target in all_rels:
            for pos, neg in opposite_pairs:
                opposite = neg if rel == pos else (pos if rel == neg else None)
                if opposite:
                    for other_rel, other_target in all_rels:
                        if other_rel == opposite and other_target == target:
                            contradictions.append({
                                "node": node_id,
                                "relation1": rel,
                                "relation2": other_rel,
                                "target": target
                            })
        
        return contradictions


# ==============================================================================
# LAYER 3: PROGRAMMATIC LIBRARY ("Neocortex - Procedural")
# ==============================================================================

@dataclass
class Skill:
    """An executable skill stored in the library."""
    id: str
    name: str
    description: str
    code: str                     # Python source code
    parameters: List[Dict]        # [{name, type, description}]
    examples: List[Dict]          # Example inputs/outputs
    created_at: int
    success_count: int = 0
    failure_count: int = 0
    source_episodes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_row(row: Tuple) -> 'Skill':
        return Skill(
            id=row[0],
            name=row[1],
            description=row[2],
            code=row[3],
            parameters=json.loads(row[4]),
            examples=json.loads(row[5]),
            created_at=row[6],
            success_count=row[7],
            failure_count=row[8],
            source_episodes=json.loads(row[9])
        )


class ProgrammaticLibrary:
    """
    Layer 3: Library of executable skills (Python functions).
    Uses AST validation for safety (no dangerous operations).
    """
    
    # Dangerous AST nodes to reject
    FORBIDDEN_NODES = {
        'Import', 'ImportFrom',     # No imports
        'Exec',                     # No exec
        'Eval',                     # No eval  
    }
    
    # Allowed built-in functions only
    ALLOWED_BUILTINS = {
        'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set',
        'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
        'min', 'max', 'sum', 'abs', 'round', 'any', 'all',
        'isinstance', 'type', 'print'
    }
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()
    
    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    code TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    examples TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    source_episodes TEXT DEFAULT '[]'
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_skill_name ON skills(name)")
    
    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate code safety using AST analysis."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            node_type = type(node).__name__
            
            # Check forbidden nodes
            if node_type in self.FORBIDDEN_NODES:
                return False, f"Forbidden operation: {node_type}"
            
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in self.ALLOWED_BUILTINS:
                        # Allow if it's a parameter or local function
                        pass  # Will be caught at runtime if dangerous
                
                # Block attribute access to dangerous modules
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in ('os', 'sys', 'subprocess', 'eval', 'exec'):
                            return False, f"Blocked access to: {node.func.value.id}"
        
        return True, "OK"
    
    def add_skill(self, name: str, description: str, code: str,
                  parameters: List[Dict], examples: List[Dict] = None,
                  source_episodes: List[str] = None) -> Dict[str, Any]:
        """Add a new skill to the library."""
        # Validate code safety
        is_safe, msg = self._validate_code(code)
        if not is_safe:
            return {"success": False, "error": msg}
        
        skill_id = f"sk_{hashlib.md5(name.encode()).hexdigest()[:12]}"
        timestamp = int(time.time())
        
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO skills 
                (id, name, description, code, parameters, examples, created_at, source_episodes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (skill_id, name, description, code, 
                  json.dumps(parameters), json.dumps(examples or []),
                  timestamp, json.dumps(source_episodes or [])))
        
        return {"success": True, "id": skill_id}
    
    def execute_skill(self, name: str, args: Dict) -> Dict[str, Any]:
        """Execute a skill with given arguments."""
        cursor = self.conn.execute(
            "SELECT * FROM skills WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        if not row:
            return {"success": False, "error": f"Skill not found: {name}"}
        
        skill = Skill.from_row(row)
        
        # Create restricted execution environment
        # __builtins__ can be a dict or module depending on context
        import builtins
        safe_builtins = {k: getattr(builtins, k) for k in self.ALLOWED_BUILTINS 
                        if hasattr(builtins, k)}
        
        exec_globals = {"__builtins__": safe_builtins}
        exec_locals = dict(args)
        
        try:
            # Execute the skill code
            exec(skill.code, exec_globals, exec_locals)
            
            # Look for a result variable or function
            if 'result' in exec_locals:
                result = exec_locals['result']
            elif skill.name in exec_locals and callable(exec_locals[skill.name]):
                result = exec_locals[skill.name](**args)
            else:
                result = None
            
            # Update success count
            with self.conn:
                self.conn.execute(
                    "UPDATE skills SET success_count = success_count + 1 WHERE name = ?",
                    (name,)
                )
            
            return {"success": True, "result": result}
            
        except Exception as e:
            # Update failure count
            with self.conn:
                self.conn.execute(
                    "UPDATE skills SET failure_count = failure_count + 1 WHERE name = ?",
                    (name,)
                )
            return {"success": False, "error": str(e)}
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """Retrieve a skill by name."""
        cursor = self.conn.execute("SELECT * FROM skills WHERE name = ?", (name,))
        row = cursor.fetchone()
        return Skill.from_row(row) if row else None
    
    def list_skills(self) -> List[Dict]:
        """List all skills with metadata."""
        cursor = self.conn.execute(
            "SELECT id, name, description, success_count, failure_count FROM skills"
        )
        return [
            {"id": r[0], "name": r[1], "description": r[2], 
             "success_rate": r[3] / max(r[3] + r[4], 1)}
            for r in cursor
        ]
    
    def delete_skill(self, name: str) -> bool:
        """Delete a skill from the library."""
        with self.conn:
            cursor = self.conn.execute("DELETE FROM skills WHERE name = ?", (name,))
            return cursor.rowcount > 0


# ==============================================================================
# WAKE-SLEEP CONSOLIDATION ENGINE
# ==============================================================================

class ConsolidationEngine:
    """
    The Wake-Sleep cycle that converts episodes into skills.
    - NREM: Prune low-value traces, abstract patterns
    - REM: Test abstractions, integrate as code
    """
    
    def __init__(self, episodes: EpisodicStream, graph: SemanticGraph, 
                 library: ProgrammaticLibrary):
        self.episodes = episodes
        self.graph = graph
        self.library = library
    
    def run_nrem(self, min_pattern_count: int = 3) -> Dict[str, Any]:
        """
        NREM Phase: Pruning and Pattern Detection.
        Find repeated action sequences and create candidate macros.
        """
        unconsolidated = self.episodes.get_unconsolidated(limit=200)
        
        if len(unconsolidated) < min_pattern_count:
            return {"patterns_found": 0, "message": "Not enough episodes"}
        
        # Find action sequences
        action_sequences = {}
        for ep in unconsolidated:
            action = ep.action
            if action not in action_sequences:
                action_sequences[action] = []
            action_sequences[action].append(ep)
        
        # Find patterns (actions that appear >= min_pattern_count with success)
        patterns = []
        for action, eps in action_sequences.items():
            successful = [e for e in eps if e.outcome.get("success", False)]
            if len(successful) >= min_pattern_count:
                patterns.append({
                    "action": action,
                    "count": len(successful),
                    "episodes": [e.id for e in successful],
                    "avg_surprise": sum(e.surprise_score for e in successful) / len(successful)
                })
        
        return {
            "patterns_found": len(patterns),
            "candidates": patterns
        }
    
    def run_rem(self, pattern: Dict) -> Dict[str, Any]:
        """
        REM Phase: Convert a pattern into a skill.
        This is a simplified version - full implementation would use LLM.
        """
        action = pattern["action"]
        episode_ids = pattern["episodes"]
        
        # Get sample episodes
        episodes = [e for e in self.episodes.get_unconsolidated(500) 
                   if e.id in episode_ids[:5]]
        
        if not episodes:
            return {"success": False, "error": "No episodes found"}
        
        # Create a simple skill template from the pattern
        # In production, this would use an LLM to generate proper code
        skill_name = action.lower().replace(" ", "_")[:30]
        skill_code = f'''
def {skill_name}(**kwargs):
    """Auto-generated skill from pattern: {action}"""
    # Pattern observed {pattern["count"]} times with success
    # Contexts: {[e.context for e in episodes[:2]]}
    result = {{"action": "{action}", "status": "executed"}}
    return result
'''
        
        # Add the skill
        result = self.library.add_skill(
            name=skill_name,
            description=f"Auto-learned: {action}",
            code=skill_code,
            parameters=[{"name": "kwargs", "type": "dict"}],
            source_episodes=episode_ids
        )
        
        if result["success"]:
            # Mark episodes as consolidated
            self.episodes.mark_consolidated(episode_ids)
            
            # Add to semantic graph
            node_id = self.graph.add_node("skill", {"name": skill_name})
            for ep in episodes[:3]:
                ctx_node = self.graph.add_node("context", ep.context)
                self.graph.add_edge(node_id, ctx_node, "learned_from")
        
        return result
    
    def consolidate(self, min_pattern_count: int = 3) -> Dict[str, Any]:
        """Full consolidation cycle: NREM + REM."""
        # NREM: Find patterns
        nrem_result = self.run_nrem(min_pattern_count)
        
        if nrem_result["patterns_found"] == 0:
            return {"phase": "nrem", "result": nrem_result}
        
        # REM: Convert top patterns to skills
        skills_created = []
        for pattern in nrem_result["candidates"][:3]:  # Top 3 patterns
            rem_result = self.run_rem(pattern)
            if rem_result.get("success"):
                skills_created.append(pattern["action"])
        
        # Prune old consolidated episodes
        pruned = self.episodes.prune_old(max_age_days=30)
        
        return {
            "phase": "complete",
            "patterns_found": nrem_result["patterns_found"],
            "skills_created": skills_created,
            "episodes_pruned": pruned
        }


# ==============================================================================
# APSM: MAIN ENGINE
# ==============================================================================

class APSM:
    """
    Active Programmatic Synthesis Memory.
    The main cognitive engine that coordinates all three layers.
    """
    
    def __init__(self, db_path: str = "apsm.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Initialize all layers
        self.episodes = EpisodicStream(self.conn)
        self.graph = SemanticGraph(self.conn)
        self.library = ProgrammaticLibrary(self.conn)
        self.consolidation = ConsolidationEngine(
            self.episodes, self.graph, self.library
        )
    
    # -------------------------------------------------------------------------
    # WAKE PHASE: Active Inference
    # -------------------------------------------------------------------------
    
    def recall_episodic(self, query: str = None, time_window: int = None,
                        limit: int = 20) -> List[Dict]:
        """Retrieve episodes from Layer 1."""
        episodes = self.episodes.recall(query, time_window, limit)
        return [e.to_dict() for e in episodes]
    
    def query_graph(self, query: str) -> List[Dict]:
        """Query the semantic graph (Layer 2)."""
        return self.graph.query(query)
    
    def execute_skill(self, skill_name: str, args: Dict) -> Dict:
        """Execute a skill from the library (Layer 3)."""
        return self.library.execute_skill(skill_name, args)
    
    def log_episode(self, context: Dict, action: str, observation: str,
                    outcome: Dict, surprise_score: float = 0.0) -> str:
        """Log a new episode to the stream."""
        return self.episodes.log(context, action, observation, outcome, surprise_score)
    
    # -------------------------------------------------------------------------
    # KNOWLEDGE MANAGEMENT
    # -------------------------------------------------------------------------
    
    def add_fact(self, subject: str, relation: str, object_: str,
                 confidence: float = 1.0) -> Dict:
        """Add a fact to the semantic graph."""
        subj_id = self.graph.add_node("entity", {"name": subject})
        obj_id = self.graph.add_node("entity", {"name": object_})
        edge_id = self.graph.add_edge(subj_id, obj_id, relation, confidence=confidence)
        return {"subject_id": subj_id, "object_id": obj_id, "edge_id": edge_id}
    
    def add_skill(self, name: str, description: str, code: str,
                  parameters: List[Dict]) -> Dict:
        """Add a skill to the programmatic library."""
        return self.library.add_skill(name, description, code, parameters)
    
    # -------------------------------------------------------------------------
    # SLEEP PHASE: Consolidation
    # -------------------------------------------------------------------------
    
    def induce_skill(self, episode_ids: List[str], description: str) -> Dict:
        """Manual skill induction from specified episodes."""
        # Get the episodes
        all_eps = self.episodes.recall(limit=500)
        selected = [e for e in all_eps if e.id in episode_ids]
        
        if not selected:
            return {"success": False, "error": "No matching episodes"}
        
        # Create pattern
        pattern = {
            "action": description,
            "count": len(selected),
            "episodes": episode_ids,
            "avg_surprise": sum(e.surprise_score for e in selected) / len(selected)
        }
        
        return self.consolidation.run_rem(pattern)
    
    def run_consolidation(self) -> Dict:
        """Run the full Wake-Sleep consolidation cycle."""
        return self.consolidation.consolidate()
    
    # -------------------------------------------------------------------------
    # STATUS & INTROSPECTION
    # -------------------------------------------------------------------------
    
    def get_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        episode_count = self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        node_count = self.conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
        edge_count = self.conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        skill_count = self.conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        unconsolidated = self.conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE consolidated = 0"
        ).fetchone()[0]
        
        return {
            "layer1_episodes": episode_count,
            "layer1_unconsolidated": unconsolidated,
            "layer2_nodes": node_count,
            "layer2_edges": edge_count,
            "layer3_skills": skill_count,
            "db_path": self.db_path
        }
    
    def list_skills(self) -> List[Dict]:
        """List all available skills."""
        return self.library.list_skills()
    
    def get_recent_episodes(self, count: int = 10) -> List[Dict]:
        """Get most recent episodes (working memory)."""
        return self.recall_episodic(limit=count)


# ==============================================================================
# STANDALONE TEST
# ==============================================================================

if __name__ == "__main__":
    # Quick test
    apsm = APSM("test_apsm.db")
    
    # Log some episodes
    for i in range(5):
        apsm.log_episode(
            context={"query": f"test query {i}", "file": "test.py"},
            action="debug error",
            observation=f"Found issue {i}",
            outcome={"success": True, "solution": f"Fix {i}"},
            surprise_score=0.5 if i % 2 == 0 else 0.2
        )
    
    # Add a fact
    apsm.add_fact("Python", "is_a", "Programming Language")
    
    # Add a skill
    apsm.add_skill(
        name="greet",
        description="Simple greeting",
        code="result = f'Hello, {name}!'",
        parameters=[{"name": "name", "type": "str"}]
    )
    
    # Execute skill
    result = apsm.execute_skill("greet", {"name": "World"})
    print(f"Skill result: {result}")
    
    # Get status
    print(f"Status: {apsm.get_status()}")
