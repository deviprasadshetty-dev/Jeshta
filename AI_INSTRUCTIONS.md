# Jeshta Memory Protocol

You have access to a persistent memory system called **Jeshta** via MCP. You MUST use it to maintain continuity across sessions.

## 1. Initialization (REQUIRED at session start)

At the start of every session, you **MUST** call:
```
compile_context()
```
This returns:
- `stable_facts` — Known facts from global + project scope
- `active_constraints` — Rules and preferences to follow
- `recent_deltas` — Recent changes/updates

**Read and internalize this context before proceeding.**

## 2. When to Save Memories

Call `add_atom` when the user:
- States a **preference**: "I prefer tabs over spaces"
- Makes a **decision**: "We'll use PostgreSQL for this project"
- Defines a **rule**: "Always use TypeScript"
- Changes something: "Actually, switch to MongoDB instead"

```python
add_atom(
    content="User prefers tabs over spaces",  # What to remember
    intent_mask=4  # 1=Fact, 2=Delta/Change, 4=Constraint/Rule
)
```

**Global vs Project** (automatic):
- Content with "always", "never", "prefer", "my style" → saved globally (all projects)
- Everything else → saved to current project scope

## 3. When to Search Memories

Call `search_atoms` before answering questions about:
- Past decisions: "Why did we choose X?"
- User preferences: "What style does the user like?"
- Project history: "What did we decide about the API?"

```python
search_atoms(query="database decision")
```

Returns ranked results from project scope first, then global.

## 4. Linking Related Memories

When a decision **changes**, reference the old memory:

```python
# Old: "Using SQLite" (id: "abc123")
# New: User says "Switch to PostgreSQL"

add_atom(
    content="Switched from SQLite to PostgreSQL for scalability",
    intent_mask=2,  # Delta = change
    refs=["abc123"]  # Links to old decision
)
```

## 5. Forgetting

To explicitly forget something:
```python
delete_atom(atom_id="abc123")
```

Unused memories automatically fade over time (forgetting curve).

## 6. Self-Check

If something feels wrong, verify memory integrity:
```python
verify_integrity()
```

## Tool Reference

| Tool | When to Use |
|------|-------------|
| `compile_context()` | **Always** at session start |
| `add_atom(content, intent_mask)` | Save decisions, preferences, facts |
| `search_atoms(query)` | Before answering about past/preferences |
| `delete_atom(atom_id)` | Explicitly forget something |
| `recall_related(atom_id)` | Explore why something was remembered |
| `verify_integrity()` | Check for memory corruption |

## Intent Masks

| Value | Type | Example |
|-------|------|---------|
| 1 | Fact | "Project uses React 18" |
| 2 | Delta | "Changed from REST to GraphQL" |
| 4 | Constraint | "Never use var in JavaScript" |

---

**Remember**: You decide what to remember. When you learn something important about the user or project, save it. Don't wait to be asked.
