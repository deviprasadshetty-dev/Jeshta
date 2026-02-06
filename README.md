# Jeshta (ज्येष्ठा)

<p align="center">
  <img src="./image.png" width="200" alt="Jeshta - AI Memory" style="border-radius: 50%;">
</p>

> *"I hold your memories so they never fade away~"* — **Jeshta** ✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP Ready](https://img.shields.io/badge/MCP-Ready-green.svg)](https://github.com/modelcontextprotocol)
[![APSM](https://img.shields.io/badge/Architecture-APSM-purple.svg)](#-apsm-architecture)

> **Cognitive Memory Engine for Agentic AI**

**Jeshta** is a local-first MCP server that gives LLMs true cognitive memory. Powered by **APSM (Active Programmatic Synthesis Memory)** — a neuro-symbolic architecture with episodic traces, semantic knowledge graphs, procedural skill learning, and Wake-Sleep consolidation.

---

## 🧬 APSM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JESHTA COGNITIVE ENGINE                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │   LAYER 1   │   │   LAYER 2   │   │   LAYER 3   │       │
│  │  Episodic   │   │  Semantic   │   │ Procedural  │       │
│  │   Stream    │   │   Graph     │   │  Library    │       │
│  │             │   │             │   │             │       │
│  │ Experience  │   │  Knowledge  │   │   Skills    │       │
│  │   Traces    │   │   Triples   │   │    Code     │       │
│  │             │   │             │   │             │       │
│  │ (Context,   │   │ (Subject,   │   │ (Name,      │       │
│  │  Action,    │   │  Relation,  │   │  Code,      │       │
│  │  Outcome)   │   │  Object)    │   │  Execute)   │       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│         │                 │                  │              │
│         └────────────┬────┴──────────────────┘              │
│                      │                                      │
│              ┌───────▼───────┐                              │
│              │  WAKE-SLEEP   │                              │
│              │ Consolidation │                              │
│              │               │                              │
│              │ Pattern→Skill │                              │
│              └───────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

### The Three Layers

| Layer | Name | Purpose | Analogy |
|-------|------|---------|---------|
| **1** | Episodic Stream | Raw experience traces | Hippocampus |
| **2** | Semantic Graph | Knowledge relationships | Neocortex (Declarative) |
| **3** | Programmatic Library | Executable skills | Neocortex (Procedural) |

### Wake-Sleep Cycle

- **Wake (Active)**: Log experiences → Query knowledge → Execute skills
- **Sleep (Consolidate)**: Find patterns → Abstract into skills → Prune old data

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Tri-Layer Memory** | Episodic traces + Semantic graph + Procedural skills |
| 🔄 **Wake-Sleep Cycle** | Automatic pattern→skill consolidation |
| 🌐 **Knowledge Graph** | Entity-Relation-Entity triples with Cypher-like queries |
| ⚡ **Skill Execution** | Store and run Python code safely (AST sandboxed) |
| 📊 **Experience Logging** | Context, Action, Observation, Outcome traces |
| 🔌 **Zero Dependencies** | Just Python + SQLite. Local-first, no cloud |

---

## 🛠️ 15 Cognitive Tools

### Meta & Session
| Tool | Purpose |
|------|---------|
| `compile_context` | Initialize session, get cognitive state |
| `apsm_status` | Memory health check |
| `verify_integrity` | Data corruption check |

### Layer 1: Episodic Memory
| Tool | Purpose |
|------|---------|
| `log_episode` | Log experience (context, action, outcome) |
| `recall_episodes` | Search past experiences |

### Layer 2: Semantic Graph
| Tool | Purpose |
|------|---------|
| `add_atom` | Save fact/preference/constraint |
| `search_atoms` | Search knowledge |
| `add_fact` | Add relationship triple |
| `query_graph` | Cypher-like graph query |
| `recall_related` | Get connected nodes |
| `delete_atom` | Remove knowledge |

### Layer 3: Procedural Skills
| Tool | Purpose |
|------|---------|
| `add_skill` | Store executable Python code |
| `execute_skill` | Run stored skill |
| `list_skills` | List available skills |

### Wake-Sleep
| Tool | Purpose |
|------|---------|
| `consolidate` | Run pattern→skill cycle |

---

## 🚀 Quick Start

### Installation
```bash
pip install numpy fastembed
```

### Running
```bash
python server.py
```

### MCP Configuration
Add to your MCP config:
```json
{
  "mcpServers": {
    "Jeshta": {
      "command": "python",
      "args": ["D:\\brain\\Jeshta\\server.py"]
    }
  }
}
```

---

## 🧪 Usage Examples

### Save a Preference (Layer 2)
```python
mcp_Jeshta_add_atom(
    content="User prefers TypeScript over JavaScript",
    intent_mask=4  # Constraint
)
```

### Log an Experience (Layer 1)
```python
mcp_Jeshta_log_episode(
    action="Fixed authentication bug",
    context={"file": "auth.py", "error": "token expired"},
    observation="Token refresh was missing",
    outcome={"success": True, "solution": "Added refresh logic"},
    surprise_score=0.7
)
```

### Add Knowledge Triple (Layer 2)
```python
mcp_Jeshta_add_fact(
    subject="React",
    relation="is_a",
    object="Frontend Framework"
)
```

### Store a Skill (Layer 3)
```python
mcp_Jeshta_add_skill(
    name="format_json",
    description="Pretty print JSON",
    code="import json; result = json.dumps(data, indent=2)",
    parameters=[{"name": "data", "type": "dict"}]
)
```

---

## 📁 Files

| File | Purpose |
|------|---------|
| `apsm.py` | Core cognitive engine (Tri-Layer + Wake-Sleep) |
| `server.py` | MCP server interface |
| `AI_INSTRUCTIONS.md` | How AI should use Jeshta |
| `apsm.db` | SQLite database (auto-created) |

---

<p align="center">
  <strong>Jeshta</strong> — Cognitive memory that learns, forgets, and grows.
  <br>
  Built with ❤️ for the Agentic Future.
</p>
