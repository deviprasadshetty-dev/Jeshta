# ΔMEM: AI Protocol Instructions

**System Prompt / Model Instructions**

To ensure your AI tools (Claude, Cursor, Windsurf) effectively use ΔMEM, paste the following rules into their "System Prompt", "Custom Instructions", or ".cursorrules" file.

---

## 🧠 Memory Protocol (ΔMEM)

You have access to a persistent, event-sourced memory engine called **ΔMEM**. You MUST use it to maintain continuity across sessions.

### 1. Initialization (Start of Session)
At the start of every new task or session, you **MUST** check for improved context.
*   **Action**: Call `compile_context(scope_hash="<project_name>")`.
*   **Goal**: Retrieve `stable_facts` and `recent_deltas` to understand the project status, user preferences, and active constraints.

### 2. Learning (During Session)
When the user defines a new rule, preference, or architectural decision:
*   **Action**: Call `add_atom`.
    *   `content`: Concise description of the fact.
    *   `scope_hash`: The project identifier (or "global" for user preferences).
    *   `intent_mask`: 1 (Fact), 2 (Delta/Change), 4 (Constraint).
    *   `embedding`: Use a zero-vector `[0.0]*dim` if you cannot generate embeddings, or request the user to handle it. (Note: Server handles quantization).

### 3. Recall (Before Answering Complex Questions)
If the user asks about a specific topic (e.g., "Why did we choose SQLite?"):
*   **Action**: Call `search_atoms(query_emb=..., scope_hash="...")`.

### 4. Evolution (State Change)
If a decision changes (e.g., "Switch from Python to Rust"):
*   **Action**: Call `add_atom` with `refs=[<id_of_old_decision>]`. 
    *   Reference the OLD atom ID so the memory engine resolves the delta correctly.

---

## 🛠 Configuration (for User)

### Claude Desktop
Add to `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "delta-mem": {
      "command": "python",
      "args": ["D:\\brain\\buddi\\server.py"]
    }
  }
}
```

### Cursor / Windsurf
1. Go to **Settings > Features > MCP**.
2. Add New Server:
   *   **Name**: `delta-mem`
   *   **Type**: `stdio`
   *   **Command**: `python D:\brain\buddi\server.py`
