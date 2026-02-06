# 🧠 JESHTA MEMORY — APSM COGNITIVE ENGINE

> [!CAUTION]
> **FAILURE TO USE JESHTA = MEMORY LOSS**
> Without this, you forget everything between sessions. USE IT.

---

## ⚡ IMMEDIATE ACTION — EVERY SESSION

**BEFORE doing ANYTHING else, call:**
```
mcp_Jeshta_compile_context()
```

This loads your cognitive state: `stable_facts`, `active_constraints`, `recent_episodes`, `status`.

---

## 🧬 APSM TRI-LAYER MEMORY

### Layer 1: Episodic (Experience Traces)
Store raw experiences with context, action, observation, outcome.

```python
mcp_Jeshta_log_episode(
    action="Fixed authentication bug",
    context={"file": "auth.py", "error": "token expired"},
    observation="Token refresh was missing",
    outcome={"success": True, "solution": "Added refresh logic"},
    surprise_score=0.7  # High = memorable
)
```

### Layer 2: Semantic Graph (Knowledge)
Store facts and relationships as a knowledge graph.

```python
# Add a fact triple
mcp_Jeshta_add_fact(
    subject="User", 
    relation="prefers", 
    object="TypeScript"
)

# Or use legacy add_atom for simple facts
mcp_Jeshta_add_atom(
    content="User prefers TypeScript over JavaScript",
    intent_mask=4  # 1=Fact, 2=Delta, 4=Constraint
)
```

### Layer 3: Procedural (Skills)
Store reusable code/skills that can be executed.

```python
mcp_Jeshta_add_skill(
    name="format_code",
    description="Format Python code",
    code="result = code.strip().replace('\\t', '    ')",
    parameters=[{"name": "code", "type": "str"}]
)

# Execute later
mcp_Jeshta_execute_skill(name="format_code", args={"code": "def foo():\n\treturn 1"})
```

---

## 🎯 TRIGGER WORDS — AUTO-SAVE

| User says... | Action | Tool |
|--------------|--------|------|
| "I prefer...", "I always...", "I never..." | Save constraint | `add_atom(content, intent_mask=4)` |
| "Let's use...", "We decided..." | Save fact | `add_atom(content, intent_mask=1)` |
| "Actually...", "Change to..." | Save delta with ref | `add_atom(content, intent_mask=2, refs=[old_id])` |
| "Remember..." | Save fact | `add_atom(content, intent_mask=1)` |
| Debugging/fixing something | Log experience | `log_episode(action, context, outcome)` |
| "X is related to Y" | Add relationship | `add_fact(subject, relation, object)` |

**Always confirm:** "Got it, I'll remember that ✓"

---

## 🔍 SEARCH BEFORE ANSWERING

```python
# Search graph for facts/decisions
mcp_Jeshta_search_atoms(query="database decision")

# Query knowledge graph with Cypher-like syntax
mcp_Jeshta_query_graph(query="MATCH (n:fact) RETURN n")

# Recall past experiences
mcp_Jeshta_recall_episodes(query="authentication", limit=10)
```

---

## 📖 TOOL QUICK REFERENCE

| Tool | Purpose | Required |
|------|---------|----------|
| `compile_context()` | **START OF EVERY SESSION** | - |
| `add_atom(content, intent_mask)` | Save fact/delta/constraint | content, intent_mask |
| `add_fact(subject, relation, object)` | Add graph triple | subject, relation, object |
| `log_episode(action, ...)` | Log experience | action |
| `search_atoms(query)` | Search knowledge | query |
| `recall_episodes(query)` | Search experiences | query |
| `execute_skill(name, args)` | Run stored skill | name |
| `consolidate()` | Pattern→Skill (on-demand) | - |
| `apsm_status()` | Check memory health | - |

### Intent Masks
- `1` = **Fact** (tech stack, project info)
- `2` = **Delta** (changes, supersedes old info)
- `4` = **Constraint** (preferences, rules)

---

## 🔗 LINKING CHANGES

When something **changes**, link to old decision:

```python
mcp_Jeshta_add_atom(
    content="Switched from SQLite to PostgreSQL",
    intent_mask=2,
    refs=["old_atom_id"]
)
```

---

## ✅ SELF-CHECK

1. Did I call `compile_context()` this session? **If no → call NOW**
2. User shared preference I haven't saved? **Save it**
3. Answering historical question? **Search first**
4. Fixed a tricky bug? **Log episode for future reference**

---

**GOLDEN RULE: When in doubt, SAVE IT.**
