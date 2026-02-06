"""
Jeshta MCP Server - APSM Cognitive Memory Engine
A unified memory system with Tri-Layer architecture.
"""
import sys
import json
import logging
import traceback
import os
from typing import Any, Dict, List
from apsm import APSM

# Configure logging to stderr so we don't pollute stdout (JSON-RPC channel)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


class MCPServer:
    def __init__(self):
        # Force DB to be in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        apsm_path = os.path.join(script_dir, "apsm.db")
        self.apsm = APSM(apsm_path)
        logging.info("APSM Cognitive Engine initialized")
        
        # Auto-detect default scope from current working directory
        self.default_scope = os.path.basename(os.getcwd())
        logging.info(f"Default scope set to: {self.default_scope}")

        # APSM-only tool surface
        self.tools = {
            # Legacy compatibility (maps to APSM)
            "add_atom": self.add_atom,              # → Semantic Graph node
            "search_atoms": self.search_atoms,      # → Graph query
            "compile_context": self.compile_context, # → APSM status + recent
            "delete_atom": self.delete_atom,        # → Delete node
            "recall_related": self.recall_related,  # → Get related nodes
            "verify_integrity": self.verify_integrity,
            # APSM Tri-Layer Tools
            "log_episode": self.log_episode,
            "recall_episodes": self.recall_episodes,
            "query_graph": self.query_graph,
            "add_fact": self.add_fact,
            "execute_skill": self.execute_skill,
            "add_skill": self.add_skill_tool,
            "list_skills": self.list_skills,
            "consolidate": self.consolidate,
            "apsm_status": self.apsm_status
        }
        
        # Initialize Embedder (Optional - for future vector search)
        self.embedder = None
        try:
            from fastembed import TextEmbedding
            import io
            logging.info("Initializing FastEmbed...")
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            finally:
                sys.stdout = original_stdout
            logging.info("FastEmbed initialized.")
        except ImportError:
            logging.warning("fastembed not installed. Vector search disabled.")
        except Exception as e:
            logging.error(f"Failed to load fastembed: {e}")

    # -------------------------------------------------------------------------
    # LEGACY COMPATIBILITY LAYER (maps old tools → APSM)
    # -------------------------------------------------------------------------
    
    def add_atom(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy: Add atom → maps to Semantic Graph node."""
        content = str(args.get("content", ""))
        if not content.strip():
            raise ValueError("Content cannot be empty")
        
        intent_mask = args.get("intent_mask", 1)
        # Map intent_mask to node label
        labels = {1: "fact", 2: "delta", 4: "constraint"}
        label = labels.get(intent_mask, "memory")
        
        node_id = self.apsm.graph.add_node(
            label=label,
            properties={
                "content": content,
                "scope": args.get("scope_hash", self.default_scope),
                "confidence": args.get("confidence", 1.0)
            },
            confidence=args.get("confidence", 1.0)
        )
        
        # Handle refs as edges
        refs = args.get("refs", [])
        for ref in refs:
            self.apsm.graph.add_edge(node_id, ref, "references")
        
        return {"id": node_id}
    
    def search_atoms(self, args: Dict[str, Any]) -> List[Dict]:
        """Legacy: Search atoms → maps to Graph query + keyword filter."""
        query_text = args.get("query", "")
        intent_mask = args.get("intent_mask")
        
        # Get all nodes from graph
        if intent_mask:
            labels = {1: "fact", 2: "delta", 4: "constraint"}
            label = labels.get(intent_mask, "memory")
            results = self.apsm.graph.query(f"MATCH (n:{label}) RETURN n")
        else:
            results = self.apsm.graph.query("MATCH (n) RETURN n")
        
        # Filter by keyword if provided
        if query_text:
            query_lower = query_text.lower()
            results = [r for r in results 
                      if query_lower in str(r.get("properties", {})).lower()]
        
        # Limit results
        top_k = args.get("top_k", 10)
        return results[:top_k]
    
    def compile_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy: Compile context → APSM status + recent episodes."""
        status = self.apsm.get_status()
        recent = self.apsm.get_recent_episodes(10)
        
        # Get facts and constraints from graph
        facts = self.apsm.graph.query("MATCH (n:fact) RETURN n")
        constraints = self.apsm.graph.query("MATCH (n:constraint) RETURN n")
        
        return {
            "stable_facts": [f["properties"].get("content", "") for f in facts[:10]],
            "active_constraints": [c["properties"].get("content", "") for c in constraints[:5]],
            "recent_episodes": len(recent),
            "status": status
        }
    
    def delete_atom(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy: Delete atom → delete graph node."""
        node_id = args.get("atom_id")
        if not node_id:
            raise ValueError("Missing field: atom_id")
        
        # Delete from graph_nodes
        with self.apsm.conn:
            cursor = self.apsm.conn.execute(
                "DELETE FROM graph_nodes WHERE id = ?", (node_id,)
            )
            # Also delete related edges
            self.apsm.conn.execute(
                "DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id)
            )
        
        return {"success": cursor.rowcount > 0, "id": node_id}
    
    def recall_related(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy: Recall related → get connected nodes."""
        node_id = args.get("atom_id")
        if not node_id:
            raise ValueError("Missing field: atom_id")
        
        return self.apsm.graph.get_related(node_id)
    
    def verify_integrity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Check APSM data integrity."""
        status = self.apsm.get_status()
        
        # Check for orphan edges
        orphan_check = self.apsm.conn.execute("""
            SELECT COUNT(*) FROM graph_edges e
            WHERE NOT EXISTS (SELECT 1 FROM graph_nodes n WHERE n.id = e.source_id)
               OR NOT EXISTS (SELECT 1 FROM graph_nodes n WHERE n.id = e.target_id)
        """).fetchone()[0]
        
        return {
            "healthy": orphan_check == 0,
            "orphan_edges": orphan_check,
            "status": status
        }

    # -------------------------------------------------------------------------
    # APSM NATIVE TOOLS
    # -------------------------------------------------------------------------
    
    def log_episode(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 1: Log an experience trace."""
        if "action" not in args:
            raise ValueError("Missing field: action")
        
        ep_id = self.apsm.log_episode(
            context=args.get("context", {}),
            action=args["action"],
            observation=args.get("observation", ""),
            outcome=args.get("outcome", {}),
            surprise_score=args.get("surprise_score", 0.0)
        )
        return {"id": ep_id, "layer": "episodic"}
    
    def recall_episodes(self, args: Dict[str, Any]) -> List[Dict]:
        """Layer 1: Recall episodes."""
        return self.apsm.recall_episodic(
            query=args.get("query"),
            time_window=args.get("time_window"),
            limit=args.get("limit", 20)
        )
    
    def query_graph(self, args: Dict[str, Any]) -> List[Dict]:
        """Layer 2: Query semantic graph."""
        query = args.get("query", "MATCH (n) RETURN n")
        return self.apsm.query_graph(query)
    
    def add_fact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 2: Add fact triple."""
        for f in ["subject", "relation", "object"]:
            if f not in args:
                raise ValueError(f"Missing field: {f}")
        
        return self.apsm.add_fact(
            subject=args["subject"],
            relation=args["relation"],
            object_=args["object"],
            confidence=args.get("confidence", 1.0)
        )
    
    def execute_skill(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 3: Execute skill."""
        if "name" not in args:
            raise ValueError("Missing field: name")
        return self.apsm.execute_skill(args["name"], args.get("args", {}))
    
    def add_skill_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 3: Add skill."""
        for f in ["name", "description", "code"]:
            if f not in args:
                raise ValueError(f"Missing field: {f}")
        
        return self.apsm.add_skill(
            name=args["name"],
            description=args["description"],
            code=args["code"],
            parameters=args.get("parameters", [])
        )
    
    def list_skills(self, args: Dict[str, Any]) -> List[Dict]:
        """Layer 3: List skills."""
        return self.apsm.list_skills()
    
    def consolidate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run Wake-Sleep consolidation."""
        return self.apsm.run_consolidation()
    
    def apsm_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get APSM status."""
        return self.apsm.get_status()

    # -------------------------------------------------------------------------
    # MCP PROTOCOL HANDLER
    # -------------------------------------------------------------------------
    
    def run(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                continue
            
            try:
                response = self.handle_request(request)
                if response:
                    print(json.dumps(response), flush=True)
            except Exception as e:
                logging.error(f"Error handling request: {e}\n{traceback.format_exc()}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {"code": -32603, "message": str(e)}
                }
                print(json.dumps(error_response), flush=True)

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        response = {"jsonrpc": "2.0", "id": request_id}
        
        if method == "initialize":
            response["result"] = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "jeshta-apsm", "version": "2.0.0"}
            }
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            response["result"] = {"tools": self._get_tool_schemas()}
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            
            if tool_name in self.tools:
                result_data = self.tools[tool_name](tool_args)
                response["result"] = {
                    "content": [{"type": "text", "text": json.dumps(result_data, default=str)}]
                }
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        elif method == "resources/list":
            response["result"] = {"resources": [
                {"uri": "jeshta://instructions", "name": "AI Instructions", 
                 "description": "How AI should use Jeshta memory", "mimeType": "text/markdown"}
            ]}
        elif method == "resources/read":
            uri = params.get("uri", "")
            if uri == "jeshta://instructions":
                script_dir = os.path.dirname(os.path.abspath(__file__))
                instructions_path = os.path.join(script_dir, "AI_INSTRUCTIONS.md")
                with open(instructions_path, "r") as f:
                    content = f.read()
                response["result"] = {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": content}]}
            else:
                raise ValueError(f"Unknown resource: {uri}")
        elif method == "prompts/list":
            response["result"] = {"prompts": []}
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return response

    def _get_tool_schemas(self) -> List[Dict]:
        return [
            # Legacy compatibility
            {"name": "add_atom", "description": "Add a new memory atom",
             "inputSchema": {"type": "object", "properties": {
                 "content": {"type": "string"}, "intent_mask": {"type": "integer"},
                 "scope_hash": {"type": "string"}, "refs": {"type": "array", "items": {"type": "string"}},
                 "confidence": {"type": "number"}
             }, "required": ["content", "intent_mask"]}},
            {"name": "search_atoms", "description": "Search for relevant atoms",
             "inputSchema": {"type": "object", "properties": {
                 "query": {"type": "string"}, "intent_mask": {"type": "integer"}, "top_k": {"type": "integer"}
             }, "required": []}},
            {"name": "compile_context", "description": "Compile active state for session",
             "inputSchema": {"type": "object", "properties": {"scope_hash": {"type": "string"}}, "required": []}},
            {"name": "delete_atom", "description": "Delete a specific atom",
             "inputSchema": {"type": "object", "properties": {"atom_id": {"type": "string"}}, "required": ["atom_id"]}},
            {"name": "recall_related", "description": "Explore causal relationships",
             "inputSchema": {"type": "object", "properties": {"atom_id": {"type": "string"}}, "required": ["atom_id"]}},
            {"name": "verify_integrity", "description": "Check for data integrity",
             "inputSchema": {"type": "object", "properties": {}, "required": []}},
            # APSM Layer 1: Episodic
            {"name": "log_episode", "description": "Layer 1: Log experience trace",
             "inputSchema": {"type": "object", "properties": {
                 "action": {"type": "string"}, "context": {"type": "object"},
                 "observation": {"type": "string"}, "outcome": {"type": "object"},
                 "surprise_score": {"type": "number"}
             }, "required": ["action"]}},
            {"name": "recall_episodes", "description": "Layer 1: Recall past episodes",
             "inputSchema": {"type": "object", "properties": {
                 "query": {"type": "string"}, "time_window": {"type": "integer"}, "limit": {"type": "integer"}
             }, "required": []}},
            # APSM Layer 2: Semantic Graph
            {"name": "query_graph", "description": "Layer 2: Query knowledge graph",
             "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}},
            {"name": "add_fact", "description": "Layer 2: Add fact triple",
             "inputSchema": {"type": "object", "properties": {
                 "subject": {"type": "string"}, "relation": {"type": "string"},
                 "object": {"type": "string"}, "confidence": {"type": "number"}
             }, "required": ["subject", "relation", "object"]}},
            # APSM Layer 3: Procedural
            {"name": "execute_skill", "description": "Layer 3: Execute stored skill",
             "inputSchema": {"type": "object", "properties": {
                 "name": {"type": "string"}, "args": {"type": "object"}
             }, "required": ["name"]}},
            {"name": "add_skill", "description": "Layer 3: Add executable skill",
             "inputSchema": {"type": "object", "properties": {
                 "name": {"type": "string"}, "description": {"type": "string"},
                 "code": {"type": "string"}, "parameters": {"type": "array"}
             }, "required": ["name", "description", "code"]}},
            {"name": "list_skills", "description": "Layer 3: List all skills",
             "inputSchema": {"type": "object", "properties": {}, "required": []}},
            # APSM Meta
            {"name": "consolidate", "description": "Run Wake-Sleep consolidation",
             "inputSchema": {"type": "object", "properties": {}, "required": []}},
            {"name": "apsm_status", "description": "Get APSM cognitive engine status",
             "inputSchema": {"type": "object", "properties": {}, "required": []}}
        ]


if __name__ == "__main__":
    server = MCPServer()
    server.run()
