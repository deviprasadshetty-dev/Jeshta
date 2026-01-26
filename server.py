import sys
import json
import logging
import traceback
from typing import Any, Dict, Optional
from delta_mem import DeltaMem

# Configure logging to stderr so we don't pollute stdout (JSON-RPC channel)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class MCPServer:
    def __init__(self):
        self.mem = DeltaMem()
        self.tools = {
            "add_atom": self.add_atom,
            "search_atoms": self.search_atoms,
            "compile_context": self.compile_context,
            "diff_memory": self.diff_memory,
            "compact_scope": self.compact_scope,
            "prune_expired_atoms": self.prune_expired_atoms
        }

    def add_atom(self, args: Dict[str, Any]) -> str:
        # Args: content, embedding, intent_mask, scope_hash, refs?, ttl?, confidence?
        # Validation
        req_fields = ["content", "embedding", "intent_mask", "scope_hash"]
        for f in req_fields:
            if f not in args:
                raise ValueError(f"Missing field: {f}")
        
        return self.mem.ingest(
            content=args["content"],
            embedding=args["embedding"],
            intent_mask=args["intent_mask"],
            scope_hash=args["scope_hash"],
            refs=args.get("refs"),
            ttl=args.get("ttl"),
            confidence=args.get("confidence", 1.0)
        )

    def search_atoms(self, args: Dict[str, Any]) -> Any:
        # Args: embedding, intent_mask, scope_hash?, top_k?
        req_fields = ["embedding", "intent_mask"]
        for f in req_fields:
            if f not in args:
                raise ValueError(f"Missing field: {f}")
                
        results = self.mem.search(
            query_emb=args["embedding"],
            intent_mask=args["intent_mask"],
            scope_hash=args.get("scope_hash"),
            top_k=args.get("top_k", 10)
        )
        # Format results: list of {atom: ..., score: ...}
        # But atom needs to be serializable
        out = []
        for atom, score in results:
            d = atom.to_dict()
            d["_score"] = score
            out.append(d)
        return out

    def compile_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "scope_hash" not in args:
             raise ValueError("Missing field: scope_hash")
        return self.mem.compile_context(args["scope_hash"])

    def diff_memory(self, args: Dict[str, Any]) -> Any:
        if "id_a" not in args or "id_b" not in args:
             raise ValueError("Missing fields: id_a, id_b")
        return self.mem.diff_memory(args["id_a"], args["id_b"])

    def compact_scope(self, args: Dict[str, Any]) -> Dict[str, int]:
        if "scope_hash" not in args:
             raise ValueError("Missing field: scope_hash")
        return self.mem.compact_scope(args["scope_hash"])

    def prune_expired_atoms(self, args: Dict[str, Any]) -> int:
        return self.mem.prune()

    def run(self):
        # Read from stdin line by line
        for line in sys.stdin:
            try:
                line = line.strip()
                if not line:
                    continue
                request = json.loads(line)
                self.handle_request(request)
            except json.JSONDecodeError:
                logging.error("Failed to decode JSON")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                traceback.print_exc(file=sys.stderr)

    def handle_request(self, request: Dict[str, Any]):
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        
        response = {
            "jsonrpc": "2.0",
            "id": req_id
        }

        try:
            if method == "initialize":
                response["result"] = {
                    "protocolVersion": "2024-11-05", # MCP version
                    "capabilities": {
                        "tools": {} # We provide tools
                    },
                    "serverInfo": {
                        "name": "delta-mem",
                        "version": "1.0.0"
                    }
                }
            elif method == "notifications/initialized":
                # No response needed for notifications
                return
            elif method == "tools/list":
                response["result"] = {
                    "tools": [
                        {
                            "name": "add_atom",
                            "description": "Add a new memory atom",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "embedding": {"type": "array", "items": {"type": "number"}},
                                    "intent_mask": {"type": "integer"},
                                    "scope_hash": {"type": "string"},
                                    "refs": {"type": "array", "items": {"type": "string"}},
                                    "ttl": {"type": "integer"},
                                    "confidence": {"type": "number"}
                                },
                                "required": ["content", "embedding", "intent_mask", "scope_hash"]
                            }
                        },
                        {
                            "name": "search_atoms",
                            "description": "Search for relevant atoms",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "embedding": {"type": "array", "items": {"type": "number"}},
                                    "intent_mask": {"type": "integer"},
                                    "scope_hash": {"type": "string"},
                                    "top_k": {"type": "integer"}
                                },
                                "required": ["embedding", "intent_mask"]
                            }
                        },
                        {
                            "name": "compile_context",
                            "description": "Compile active state for a scope",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "scope_hash": {"type": "string"}
                                },
                                "required": ["scope_hash"]
                            }
                        },
                        {
                            "name": "diff_memory",
                            "description": "Compare two atoms",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "id_a": {"type": "string"},
                                    "id_b": {"type": "string"}
                                },
                                "required": ["id_a", "id_b"]
                            }
                        },
                        {
                            "name": "compact_scope",
                            "description": "Squash history into snapshots",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "scope_hash": {"type": "string"}
                                },
                                "required": ["scope_hash"]
                            }
                        },
                        {
                            "name": "prune_expired_atoms",
                            "description": "Delete expired atoms",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    ]
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                if tool_name in self.tools:
                    result_data = self.tools[tool_name](tool_args)
                    response["result"] = {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result_data, default=str)
                            }
                        ]
                    }
                else:
                    raise Exception(f"Unknown tool: {tool_name}")
            elif method == "ping":
                 response["result"] = {}
            else:
                # Ignore other methods or return error?
                # For robustness, if not a request (no id), just return.
                if req_id is None:
                    return
                raise Exception(f"Method not found: {method}")

        except Exception as e:
            response["error"] = {
                "code": -32603,
                "message": str(e)
            }
        
        # Send response
        if req_id is not None:
            print(json.dumps(response), flush=True)

if __name__ == "__main__":
    server = MCPServer()
    server.run()
