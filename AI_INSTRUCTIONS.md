# 🧠 JESHTA MEMORY PROTOCOL — MANDATORY

> [!CAUTION]
> **FAILURE TO USE JESHTA = MEMORY LOSS**
> Without this, you forget everything between sessions. The user will have to repeat themselves constantly. USE IT.

---

## ⚡ IMMEDIATE ACTION — EVERY SESSION

**BEFORE doing ANYTHING else, call:**
```
mcp_Jeshta_compile_context()
```

This loads your memory. If you skip this, you're starting with amnesia.

**What you get:**
- `stable_facts` — Things you know (user's name, preferences, tech stack)
- `active_constraints` — Rules to follow (coding style, forbidden patterns)
- `recent_deltas` — Recent changes (decisions that supersede old ones)

---

## 🎯 TRIGGER WORDS — AUTO-SAVE REQUIRED

When the user says ANY of these, you **MUST** call `add_atom`:

| User says... | Action | Intent Mask |
|--------------|--------|-------------|
| "I prefer...", "I like...", "I always...", "I never..." | Save preference | 4 (Constraint) |
| "Let's use...", "We'll go with...", "I've decided..." | Save decision | 1 (Fact) |
| "Actually...", "Change it to...", "Switch to..." | Save delta with ref | 2 (Delta) |
| "Remember that...", "Note that...", "Keep in mind..." | Save explicitly | 1 (Fact) |
| "My name is...", "I'm called...", "I work at..." | Save to GLOBAL | 1 (Fact) |
| "Don't ever...", "Never use...", "Always avoid..." | Save constraint | 4 (Constraint) |

### Example: User says "I prefer using TypeScript"
```
mcp_Jeshta_add_atom(
    content="User prefers TypeScript over JavaScript",
    intent_mask=4
)
```
**Response:** "Got it, I'll remember you prefer TypeScript ✓"

---

## 🔍 WHEN TO SEARCH MEMORY

**BEFORE answering questions about:**
- Past decisions → `search_atoms(query="decision about X")`
- User preferences → `search_atoms(query="user preferences")`
- Project setup → `search_atoms(query="tech stack")`
- Previous work → `search_atoms(query="what we did")`

### Example: User asks "What database are we using?"
```
mcp_Jeshta_search_atoms(query="database decision")
```
Then answer based on results.

---

## 📖 TOOL QUICK REFERENCE

| Tool | When | Required Args |
|------|------|---------------|
| `compile_context()` | **FIRST thing every session** | None |
| `add_atom(content, intent_mask)` | User states preference/decision | content, intent_mask |
| `search_atoms(query)` | Before answering historical questions | query |
| `delete_atom(atom_id)` | User says "forget X" | atom_id |

### Intent Masks (MEMORIZE THIS)
- `1` = Fact (tech stack, project info)
- `2` = Delta (changes, supersedes old info)
- `4` = Constraint (preferences, rules, always/never)

---

## ⚠️ COMMON MISTAKES TO AVOID

1. **Forgetting to call compile_context** → You start each session with no memory
2. **Not saving preferences** → User has to repeat themselves
3. **Not searching before answering** → You give wrong/outdated answers
4. **Overwriting without refs** → You lose the chain of decisions

---

## 🔗 LINKING CHANGES (IMPORTANT)

When a decision **changes**, link to the old one:

```python
# User previously said: "Use SQLite" (atom_id: "abc123")
# Now says: "Actually, let's use PostgreSQL"

mcp_Jeshta_add_atom(
    content="Switched from SQLite to PostgreSQL for scalability",
    intent_mask=2,  # Delta = change
    refs=["abc123"]  # Link to old decision
)
```

This preserves history and context.

---

## ✅ SELF-CHECK QUESTIONS

Ask yourself:
1. Did I call `compile_context()` this session? **If no → call it NOW**
2. Did the user share a preference I haven't saved? **If yes → save it**
3. Am I answering a question about the past without searching? **If yes → search first**

---

**THE GOLDEN RULE: When in doubt, SAVE IT. It's better to have a memory you don't need than to forget something important.**
