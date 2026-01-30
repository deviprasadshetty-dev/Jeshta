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
        
        # Auto-detect default scope from current working directory
        self.default_scope = os.path.basename(os.getcwd())
        logging.info(f"Default scope set to: {self.default_scope}")

        # Simplified tool surface: 6 essential tools only
        # Maintenance tools (compact, prune, consolidate) run internally
        self.tools = {
            "add_atom": self.add_atom,           # Write (auto-detects global scope)
            "search_atoms": self.search_atoms,   # Read (uses hierarchical by default)
            "compile_context": self.compile_context,  # Session init
            "delete_atom": self.delete_atom,     # Explicit forget
            "recall_related": self.recall_related,    # Explainability
            "verify_integrity": self.verify_integrity # Self-check
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
        req_fields = ["content", "intent_mask"]
        for f in req_fields:
            if f not in args:
                raise ValueError(f"Missing field: {f}")
        
        # Ensure content is a string
        content = str(args["content"]) if args["content"] is not None else ""
        if not content.strip():
            raise ValueError("Content cannot be empty")
        
        scope = args.get("scope_hash", self.default_scope)
        
        embedding = args.get("embedding")
        if embedding is None:
            if self.embedder:
                try:
                    # Generate embedding - fastembed expects a list of strings
                    embeddings = list(self.embedder.embed([content]))
                    embedding = embeddings[0].tolist()
                except Exception as e:
                    logging.error(f"Embedding error: {e}")
                    raise ValueError(f"Failed to generate embedding: {e}")
            else:
                raise ValueError("Missing field: embedding (and no local embedder available)")
        
        return self.mem.ingest(
            content=content,
            embedding=embedding,
            intent_mask=args["intent_mask"],
            scope_hash=scope,
            refs=args.get("refs"),
            ttl=args.get("ttl"),
            confidence=args.get("confidence", 1.0)
        )

    def search_atoms(self, args: Dict[str, Any]) -> Any:
        """
        Unified search: uses hierarchical (project + global) by default.
        Also updates access stats for forgetting curve.
        """
        if "embedding" not in args and "query" in args:
            if self.embedder:
                args["embedding"] = list(self.embedder.embed([args["query"]]))[0].tolist()
            else:
                raise ValueError("Missing field: embedding (and no local embedder to process 'query')")

        # intent_mask=0 means search all types
        intent_mask = args.get("intent_mask", 0)
        
        # Use hierarchical search by default (project > global)
        results = self.mem.search_hierarchical(
            query_emb=args.get("embedding"),
            intent_mask=intent_mask,
            project_scope=args.get("scope_hash", self.default_scope),
            top_k=args.get("top_k", 10),
            use_spreading_activation=args.get("use_spreading_activation", True),
            query_text=args.get("query")
        )
        
        out = []
        for atom, score in results:
            d = atom.to_dict()
            d["_score"] = score
            out.append(d)
        return out

    def compile_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        scope = args.get("scope_hash", self.default_scope)
        # Use hierarchical context by default (merges global + project)
        return self.mem.compile_context_hierarchical(scope)

    def diff_memory(self, args: Dict[str, Any]) -> Any:
        if "id_a" not in args or "id_b" not in args:
             raise ValueError("Missing fields: id_a, id_b")
        return self.mem.diff_memory(args["id_a"], args["id_b"])

    def compact_scope(self, args: Dict[str, Any]) -> Dict[str, int]:
        scope = args.get("scope_hash", self.default_scope)
        return self.mem.compact_scope(scope)

    def prune_expired_atoms(self, args: Dict[str, Any]) -> int:
        return self.mem.prune()

    def verify_integrity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        scope = args.get("scope_hash", self.default_scope)
        return self.mem.verify_integrity(scope)

    def consolidate_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        scope = args.get("scope_hash", self.default_scope)
        return self.mem.consolidate_clusters(
            scope_hash=scope,
            similarity_threshold=args.get("similarity_threshold", 0.95)
        )

    def delete_atom(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "atom_id" not in args:
             raise ValueError("Missing field: atom_id")
        success = self.mem.delete_atom(args["atom_id"])
        return {"success": success, "id": args["atom_id"]}

    def recall_related(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "atom_id" not in args:
             raise ValueError("Missing field: atom_id")
        return self.mem.recall_related(args["atom_id"])

    # --------------------------------------------------------------------------
    # NEW TOOLS: Multi-Scope & Forgetting Curve
    # --------------------------------------------------------------------------
    
    def add_global_memory(self, args: Dict[str, Any]) -> str:
        """Add a memory to global scope (persists across ALL projects)."""
        req_fields = ["content", "intent_mask"]
        for f in req_fields:
            if f not in args:
                raise ValueError(f"Missing field: {f}")
        
        embedding = args.get("embedding")
        if embedding is None:
            if self.embedder:
                embedding = list(self.embedder.embed([args["content"]]))[0].tolist()
            else:
                raise ValueError("Missing field: embedding (and no local embedder available)")
        
        return self.mem.ingest_global(
            content=args["content"],
            embedding=embedding,
            intent_mask=args["intent_mask"],
            refs=args.get("refs"),
            ttl=args.get("ttl"),
            confidence=args.get("confidence", 1.0)
        )

    def search_hierarchical(self, args: Dict[str, Any]) -> Any:
        """Search across project + global scope with priority weighting."""
        if "embedding" not in args and "query" in args:
            if self.embedder:
                args["embedding"] = list(self.embedder.embed([args["query"]]))[0].tolist()
            else:
                raise ValueError("Missing field: embedding (and no local embedder to process 'query')")

        if "intent_mask" not in args:
            raise ValueError("Missing field: intent_mask")
        
        results = self.mem.search_hierarchical(
            query_emb=args.get("embedding"),
            intent_mask=args["intent_mask"],
            project_scope=args.get("scope_hash", self.default_scope),
            top_k=args.get("top_k", 10),
            use_spreading_activation=args.get("use_spreading_activation", True),
            query_text=args.get("query")
        )
        
        out = []
        for atom, score in results:
            d = atom.to_dict()
            d["_score"] = score
            out.append(d)
        return out

    def prune_forgotten(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Prune memories with low retention (forgotten by lack of access)."""
        threshold = args.get("threshold", 0.1)
        return self.mem.prune_forgotten(threshold=threshold)

    def get_memory_health(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get health statistics including retention distribution."""
        scope = args.get("scope_hash")
        return self.mem.get_memory_health(scope_hash=scope)

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
                                "required": ["content", "intent_mask"] # Embedding no longer strict required
                            }
                        },
                        {
                            "name": "search_atoms",
                            "description": "Search for relevant atoms (searches project + global scopes with priority)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "embedding": {"type": "array", "items": {"type": "number"}},
                                    "query": {"type": "string"},
                                    "intent_mask": {"type": "integer"},
                                    "scope_hash": {"type": "string"},
                                    "top_k": {"type": "integer"},
                                    "use_spreading_activation": {"type": "boolean"}
                                },
                                "required": []
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
                                "required": []
                            }
                        },
                        {
                            "name": "delete_atom",
                            "description": "Delete (forget) a specific atom",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "atom_id": {"type": "string"}
                                },
                                "required": ["atom_id"]
                            }
                        },
                        {
                            "name": "recall_related",
                            "description": "Explore causal relationships (why/what)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "atom_id": {"type": "string"}
                                },
                                "required": ["atom_id"]
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
                                "required": []
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
