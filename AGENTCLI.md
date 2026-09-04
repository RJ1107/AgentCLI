# AGENTCLI.md

This file is the project-level long-term context for AgentCLI. It should contain durable engineering rules and stable architecture notes only. Temporary debugging notes, one-off task state, and private preferences do not belong here.

## Priority

1. Actual source code and tests
2. `AGENTS.md`, if present
3. This file, `AGENTCLI.md`
4. `README.md`
5. Documentation under `docs/`

Documentation must match behavior exposed through real public entry points. Do not claim a capability just because a prompt, README section, or test fixture mentions it.

## Project

- Name: AgentCLI Python
- Purpose: terminal-native AI coding-agent CLI with file tools, command execution, MCP integration, memory, snapshots, Runtime API, and background task support.
- Python: 3.11+
- Package manager: `uv`
- Primary entry points: interactive REPL, single-prompt CLI, SDK, MCP server, Runtime API, and task worker.

## Common Commands

```bash
uv sync --extra dev
uv run agentcli
uv run agentcli -p "Explain this repository"
uv run agentcli --mode plan -p "Plan first, then execute"
uv run agentcli --mode team --worker-mode plan -p "Review core modules in parallel"
uv run agentcli doctor --cwd .
```

Verification:

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev python -m pytest
uv build
uv run agentcli --version
uv run agentcli doctor --cwd .
```

Prefer `uv run --extra dev python -m pytest` in this environment; do not assume bare `pytest` is available.

## Architecture

The three execution paths share the same tool registry, safety policy, memory, skill loading, and snapshot services.

| Path | Main module | Trigger |
| --- | --- | --- |
| ReAct | `agent/agent.py` | default REPL or `-p` |
| Plan-and-Execute | `agent/plan_execute.py` | `/plan` or `--mode plan` |
| Multi-Agent | `agent/orchestrator.py` | `/team` or `--mode team` |

Core modules:

```text
src/agentcli/
├── agent/       ReAct, Plan-and-Execute, Multi-Agent orchestration
├── entrypoints/ Typer CLI and prompt-toolkit REPL
├── llm/         model abstraction, OpenAI-compatible client, usage/cost
├── tools/       tool definitions, executor, file and command operations
├── policy/      PathGuard, CommandGuard, AuditLog
├── prompt/      static prompt and request-specific dynamic context
├── context/     context budgeting and compression
├── memory/      SQLite long-term memory, recall, and governance
├── skill/       builtin/user/project skill discovery and injection
├── plan/        plan models, tasks, and DAG execution
├── mcp/         MCP client, server, and dynamic tool registration
├── runtime/     thread/turn API and durable task queue
├── render/      Rich and plain terminal rendering
├── snapshot/    pre/post run snapshots and restore
└── rag/         local code indexing and search
```

Use `rg` for exact code location. RAG `search_code` is a semantic assistant, not a replacement for precise text search.

## Change Coupling

- CLI or slash command changes: check `entrypoints/cli.py`, `entrypoints/repl.py`, README, and tests.
- Agent behavior changes: verify ReAct, Plan, Team, SDK, and Runtime entry points.
- Tool changes: check schema, executor behavior, read/write flags, concurrency, safety policy, audit logs, and tests.
- HITL or permission-mode changes: verify approval callbacks, PathGuard, CommandGuard, MCP write tools, terminal status, and default-policy restoration.
- Model protocol changes: verify factory, streaming events, tool calls, reasoning fields, usage accounting, price profiles, and docs.
- Memory, Prompt, or Skill changes: keep static context, dynamic recall, and one-shot skill injection boundaries clear.
- Runtime task changes: preserve project scope, atomic claiming, lease recovery, and cancellation-safe completion.

## Safety

- Default mode keeps HITL, workspace path restrictions, command guard, and audit logging enabled.
- Interactive `Auto (full access)` is a user-selected session mode that disables HITL/path/command guards until switched back.
- Non-interactive entry points must not silently bypass dangerous operations because no approval UI is available.
- Do not commit `.env`, real API keys, tokens, cookies, private keys, user-level databases, or audit logs.
- Logs, exceptions, and test output must not leak secrets.
- Do not use destructive Git commands to overwrite user changes.

## Known Boundaries

- Real LLM calls require a valid API key.
- Chrome DevTools MCP requires compatible Node.js, npm/npx, and Chrome.
- Runtime turn execution needs working model configuration.
- Web search and fetch are best-effort public HTTP tools; browser-login state belongs in Chrome DevTools MCP or another dedicated connector.
