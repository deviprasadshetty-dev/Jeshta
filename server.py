import sys
import json
import logging
import traceback
import os
from typing import Any, Dict, Optional
from delta_mem import DeltaMem

# Configure logging to stderr so we don't pollute stdout (JSON-RPC channel)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class MCPServer:
    def __init__(self):
        # Force DB to be in the same directory as this script
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delta_mem.db")
        self.mem = DeltaMem(db_path)
        self.tools = {
            "add_atom": self.add_atom,
            "search_atoms": self.search_atoms,
            "compile_context": self.compile_context,
            "diff_memory": self.diff_memory,
            "compact_scope": self.compact_scope,
            "prune_expired_atoms": self.prune_expired_atoms,
            "verify_integrity": self.verify_integrity,
            "consolidate_memory": self.consolidate_memory
        }
        
        # Initialize Embedder (Optional)
        self.embedder = None
        try:
            from fastembed import TextEmbedding
            # Use a lightweight, high-performance model. 
            # BAAI/bge-small-en-v1.5 is excellent (384 dim).
            logging.info("Initializing FastEmbed (BAAI/bge-small-en-v1.5)...")
            self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            logging.info("FastEmbed initialized.")
        except ImportError:
            logging.warning("fastembed not installed. Auto-embedding disabled.")
        except Exception as e:
            logging.error(f"Failed to load fastembed: {e}")

    def add_atom(self, args: Dict[str, Any]) -> str:
        # Args: content, embedding, intent_mask, scope_hash, refs?, ttl?, confidence?
        # Validation
        req_fields = ["content", "intent_mask", "scope_hash"]
        for f in req_fields:
            if f not in args:
                raise ValueError(f"Missing field: {f}")
        
        embedding = args.get("embedding")
        if embedding is None:
            if self.embedder:
                # Generate embedding
                # fastembed returns a generator, so we take the first item
                embedding = list(self.embedder.embed([args["content"]]))[0].tolist()
            else:
                raise ValueError("Missing field: embedding (and no local embedder available)")
        
        return self.mem.ingest(
            content=args["content"],
            embedding=embedding,
            intent_mask=args["intent_mask"],
            scope_hash=args["scope_hash"],
            refs=args.get("refs"),
            ttl=args.get("ttl"),
            confidence=args.get("confidence", 1.0)
        )

    def search_atoms(self, args: Dict[str, Any]) -> Any:
        # Args: embedding, intent_mask, scope_hash?, top_k?
        if "embedding" not in args and "query" in args:
             # Support "query" if embedding missing
             if self.embedder:
                 args["embedding"] = list(self.embedder.embed([args["query"]]))[0].tolist()
             else:
                 raise ValueError("Missing field: embedding (and no local embedder to process 'query')")

        req_fields = ["embedding", "intent_mask"]
        for f in req_fields:
            if f not in args:
                raise ValueError(f"Missing field: {f}")
                
        results = self.mem.search(
            query_emb=args["embedding"],
            intent_mask=args["intent_mask"],
            scope_hash=args.get("scope_hash"),
            top_k=args.get("top_k", 10),
            use_spreading_activation=args.get("use_spreading_activation", True)
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

    def verify_integrity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "scope_hash" not in args:
             raise ValueError("Missing field: scope_hash")
        return self.mem.verify_integrity(args["scope_hash"])

    def consolidate_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "scope_hash" not in args:
             raise ValueError("Missing field: scope_hash")
        return self.mem.consolidate_clusters(
            scope_hash=args["scope_hash"],
            similarity_threshold=args.get("similarity_threshold", 0.95)
        )

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
                        "tools": {},
                        "resources": {},
                        "prompts": {}
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
                                "required": ["content", "intent_mask", "scope_hash"] # Embedding no longer strict required
                            }
                        },
                        {
                            "name": "search_atoms",
                            "description": "Search for relevant atoms",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "embedding": {"type": "array", "items": {"type": "number"}},
                                    "query": {"type": "string"}, # Added support for raw query
                                    "intent_mask": {"type": "integer"},
                                    "scope_hash": {"type": "string"},
                                    "scope_hash": {"type": "string"},
                                    "top_k": {"type": "integer"},
                                    "use_spreading_activation": {"type": "boolean"}
                                },
                                "required": ["intent_mask"] # Embedding/Query optional group
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
                        },
                        {
                            "name": "verify_integrity",
                            "description": "Check for graph paradoxes and corruption",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "scope_hash": {"type": "string"}
                                },
                                "required": ["scope_hash"]
                            }
                        },
                        {
                            "name": "consolidate_memory",
                            "description": "Cluster and merge similar memories (Dreaming)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "scope_hash": {"type": "string"},
                                    "similarity_threshold": {"type": "number"}
                                },
                                "required": ["scope_hash"]
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
                    
            elif method == "resources/list":
                response["result"] = {
                    "resources": [
                        {
                            "uri": "mem://stats",
                            "name": "Memory Statistics",
                            "description": "Current statistics of the Delta Memory engine",
                            "mimeType": "application/json"
                        },
                        {
                            "uriTemplate": "mem://atom/{id}",
                            "name": "Atom Inspector",
                            "description": "View details of a specific memory atom",
                            "mimeType": "application/json"
                        }
                    ]
                }
                
            elif method == "resources/read":
                uri = params.get("uri")
                if uri == "mem://stats":
                    stats = self.mem.get_stats()
                    response["result"] = {
                        "contents": [{
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(stats, indent=2)
                        }]
                    }
                elif uri.startswith("mem://atom/"):
                    atom_id = uri.replace("mem://atom/", "")
                    atoms = self.mem.db.get_atoms_by_ids([atom_id])
                    if not atoms:
                        raise Exception("Atom not found")
                    response["result"] = {
                        "contents": [{
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(atoms[0].to_dict(), default=str, indent=2)
                        }]
                    }
                else:
                    raise Exception("Resource not found")

            elif method == "prompts/list":
                response["result"] = {
                    "prompts": [
                        {
                            "name": "recall_context",
                            "description": "Recall context for a specific task",
                            "arguments": [
                                {
                                    "name": "task_description",
                                    "description": "Description of the task to recall context for",
                                    "required": True
                                },
                                {
                                    "name": "scope",
                                    "description": "Project scope hash",
                                    "required": True
                                }
                            ]
                        },
                        {
                            "name": "save_decision",
                            "description": "Save an architectural decision",
                            "arguments": [
                                {
                                    "name": "decision",
                                    "description": "The decision made",
                                    "required": True
                                },
                                {
                                    "name": "reasoning",
                                    "description": "Why this decision was made",
                                    "required": True
                                },
                                {
                                    "name": "scope",
                                    "description": "Project scope hash",
                                    "required": True
                                }
                            ]
                        }
                    ]
                }

            elif method == "prompts/get":
                prompt_name = params.get("name")
                args = params.get("arguments", {})
                
                if prompt_name == "recall_context":
                    task = args.get("task_description", "")
                    scope = args.get("scope", "global")
                    response["result"] = {
                        "description": f"Recalling context for: {task}",
                        "messages": [
                            {
                                "role": "user",
                                "content": {
                                    "type": "text",
                                    "text": f"Please checking DeltaMem for context regarding: {task}\nFirst, I will search for relevant atoms locally..."
                                }
                            },
                            {
                                "role": "assistant",
                                "content": {
                                    "type": "text",
                                    "text": "I'll help you look that up."
                                }
                            }
                            # Note: Prompts in MCP are static messages to inject.
                            # Real dynamic behavior usually involves the client seeing this and carrying it out.
                            # We can guide the user to call tools.
                        ]
                    }
                elif prompt_name == "save_decision":
                    decision = args.get("decision", "")
                    reason = args.get("reasoning", "")
                    scope = args.get("scope", "global")
                    response["result"] = {
                         "description": "Saving decision",
                         "messages": [
                             {
                                 "role": "user", 
                                 "content": {
                                     "type": "text", 
                                     "text": f"Save this decision to memory:\nDecision: {decision}\nReason: {reason}\nScope: {scope}"
                                 }
                             }
                         ]
                    }
                else:
                    raise Exception(f"Unknown prompt: {prompt_name}")

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
