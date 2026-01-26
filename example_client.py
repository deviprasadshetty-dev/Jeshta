import subprocess
import sys
import json
import time
import os
import random

def rpc_request(process, method, params, req_id):
    req = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": req_id
    }
    msg = json.dumps(req)
    process.stdin.write(msg + "\n")
    process.stdin.flush()
    
    # Read response
    while True:
        line = process.stdout.readline()
        if not line:
            raise Exception("Server closed connection")
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        if resp.get("id") == req_id:
            if "error" in resp:
                raise Exception(f"RPC Error: {resp['error']}")
            return resp.get("result")
        # Ignore notifications or other IDs

def rpc_notify(process, method, params):
    req = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params
    }
    msg = json.dumps(req)
    process.stdin.write(msg + "\n")
    process.stdin.flush()

def main():
    server_script = os.path.join(os.path.dirname(__file__), "server.py")
    
    # Start server
    print(f"Starting server: {sys.executable} {server_script}")
    process = subprocess.Popen(
        [sys.executable, server_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr, # Server logs to stderr
        text=True,
        bufsize=1
    )

    try:
        # Handshake
        print("\n--- Handshake ---")
        init_res = rpc_request(process, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "example-client", "version": "1.0"}
        }, 1)
        print("Initialized:", json.dumps(init_res, indent=2))
        
        rpc_notify(process, "notifications/initialized", {})
        print("Sent initialized notification")
        
        # Tools List
        print("\n--- Listing Tools ---") # Optional but good check
        tools_res = rpc_request(process, "tools/list", {}, 2)
        print("Tools:", [t['name'] for t in tools_res['tools']])
        
        # 1. Add Identity
        print("\n--- Adding Identity Atom ---")
        # Embedding: random 4-dim vector for demo
        emb_identity = [0.1, 0.9, 0.0, 0.0]
        
        # Call tool 'add_atom'
        res = rpc_request(process, "tools/call", {
            "name": "add_atom",
            "arguments": {
                "content": "User name is Alice.",
                "embedding": emb_identity,
                "intent_mask": 1, # Fact
                "scope_hash": "user_profile"
            }
        }, 3)
        print("Add Atom Result:", res)
        identity_atom_id = json.loads(res['content'][0]['text'])
        
        # 2. Add Preference
        print("\n--- Adding Preference Atom ---")
        emb_pref = [0.0, 0.2, 0.8, 0.0]
        res = rpc_request(process, "tools/call", {
            "name": "add_atom",
            "arguments": {
                "content": "Alice prefers Dark Mode.",
                "embedding": emb_pref,
                "intent_mask": 1, 
                "scope_hash": "user_profile"
            }
        }, 4)
        print("Add Pref Result:", res)
        pref_atom_id = json.loads(res['content'][0]['text'])
        
        # 3. Add Delta (Change Preference)
        print("\n--- Adding Delta Atom (Light Mode) ---")
        time.sleep(1.1) # Ensure time difference
        emb_delta = [0.0, 0.2, 0.85, 0.0] # Similiar to pref
        res = rpc_request(process, "tools/call", {
            "name": "add_atom",
            "arguments": {
                "content": "Alice changed preference to Light Mode.",
                "embedding": emb_delta,
                "intent_mask": 2, # Delta
                "scope_hash": "user_profile",
                "refs": [pref_atom_id] # References previous state
            }
        }, 5)
        print("Add Delta Result:", res)
        
        # 4. Compile Context
        print("\n--- Compiling Context (Should match 'Light Mode') ---")
        res = rpc_request(process, "tools/call", {
            "name": "compile_context",
            "arguments": {
                "scope_hash": "user_profile"
            }
        }, 6)
        context = json.loads(res['content'][0]['text'])
        print("Compiled Context:", json.dumps(context, indent=2))
        
        # Check correctness
        facts = context.get("stable_facts", [])
        deltas = context.get("recent_deltas", [])
        
        # "Alice changed preference to Light Mode" should be in stable or recent depending on time logic
        # Code says < 3600s is recent delta.
        print("\nVerification:")
        found_light = any("Light Mode" in s for s in deltas + facts)
        found_dark = any("Alice prefers Dark Mode" in s for s in facts + deltas)
        
        if found_light and not found_dark:
            print("SUCCESS: Light Mode resolved, Dark Mode superceded.")
        elif found_light and found_dark:
             print("PARTIAL: Both present (maybe reference logic didn't prune old?).")
        else:
            print("FAILURE: Light Mode not found.")

        # 5. Temporal Compaction
        print("\n--- Testing Temporal Compaction ---")
        res = rpc_request(process, "tools/call", {
            "name": "compact_scope",
            "arguments": {
                "scope_hash": "user_profile"
            }
        }, 7)
        print("Compact Result:", res)
        # Verify that we still see "Light Mode"
        res = rpc_request(process, "tools/call", {
            "name": "compile_context",
            "arguments": {
                "scope_hash": "user_profile"
            }
        }, 8)
        context = json.loads(res['content'][0]['text'])
        print("Post-Compact Context:", json.dumps(context, indent=2))
        
        # Check integrity
        # Should contain "Light Mode" as a SNAPSHOT fact.
        # Note: Delta logic puts "recent deltas" as deltas, but when compacted, they might become stable facts if new snapshot refs are empty.
        # Since references are cleared, "Light Mode" atom is now a root.
        # If it was a recent delta, it remains a recent delta because created_at is NOW.
        # Wait - created_at is NOW, so it's very recent.
        
        verified = any("Light Mode" in s for s in context['recent_deltas'] + context['stable_facts'])
        if verified:
             print("SUCCESS: Compaction preserved state.")
        else:
             print("FAILURE: State lost after compaction.")

        # 6. Cross-Project Search
        print("\n--- Testing Cross-Project Search ---")
        # Add atom in a DIFFERENT project
        emb_proj_b = [0.0, 0.0, 0.0, 1.0]
        rpc_request(process, "tools/call", {
            "name": "add_atom",
            "arguments": {
                "content": "Project B uses Rust.",
                "embedding": emb_proj_b,
                "intent_mask": 1,
                "scope_hash": "project_beta"
            }
        }, 9)
        
        # Search Specific Scope (Project Beta)
        res_beta = rpc_request(process, "tools/call", {
            "name": "search_atoms",
            "arguments": {
                "embedding": emb_proj_b,
                "intent_mask": 0,
                "scope_hash": "project_beta"
            }
        }, 10)
        print("Search(Project Beta): Found", len(json.loads(res_beta['content'][0]['text'])), "atoms.")
        
        # Search Global (*)
        res_global = rpc_request(process, "tools/call", {
            "name": "search_atoms",
            "arguments": {
                "embedding": emb_proj_b, # Query shouldn't matter with mask 0 mostly, but we use it.
                "intent_mask": 0,
                "scope_hash": "*"
            }
        }, 11)
        global_atoms = json.loads(res_global['content'][0]['text'])
        print("Search(*): Found", len(global_atoms), "atoms.")
        
        # Verify differentiation
        scopes = {a['scope_hash'] for a in global_atoms}
        if "user_profile" in scopes and "project_beta" in scopes:
             print("SUCCESS: Global search found atoms from multiple scopes:", scopes)
        else:
             print("FAILURE: Global search missing scopes. Got:", scopes)

    except Exception as e:
        print(f"Client failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        process.terminate()

if __name__ == "__main__":
    main()
