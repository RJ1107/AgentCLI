# AgentCLI Python

AgentCLI Python is a terminal-native AI agent CLI for working inside real software projects. It can read and edit files, search code, run approved commands, call MCP tools, keep scoped memory, create snapshots, and expose a lightweight Runtime API for thread and background task execution.

The project is designed as a practical coding-agent workbench rather than a UI-only demo. Core paths are covered by tests and can be exercised from the terminal through ReAct, plan-and-execute, and multi-agent workflows.

## Features

- Interactive terminal agent built with Rich and prompt-toolkit
- Single-prompt mode for scripting and automation
- OpenAI-compatible streaming LLM client, with DeepSeek as the default provider
- Provider-specific API key discovery for DeepSeek, GLM, Kimi, Step, and other compatible services
- ReAct loop with thinking, tool call, tool result, final output, usage, and cost events
- Plan-and-Execute mode with Planner-generated DAGs and dependency-aware execution
- Multi-Agent mode with Planner, parallel Workers, Reviewer, retry limits, and optional plan-mode workers
- Built-in file, shell, grep, glob, memory, web search, web fetch, code search, skill, and snapshot tools
- HITL confirmation, command/path safety policies, and JSONL audit logs
- MCP client for stdio and Streamable HTTP servers
- Built-in MCP server mode for exposing AgentCLI tools
- Runtime API for threads, turns, event logs, and durable background tasks
- Project-scoped SQLite memory with relevance recall, deduplication, TTL, and capacity control
- Context budget management and deterministic conversation compression
- Pre-run and post-run workspace snapshots with restore support
- Local and remote image reference parsing, with provider capability fallback

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Optional: `rg` for faster local search
- Optional: Node.js, npm/npx, and Chrome for Chrome DevTools MCP integration

## Quick Start

```bash
uv sync --extra dev
uv run agentcli --help
```

Start the interactive REPL:

```bash
uv run agentcli
```

Run a single prompt:

```bash
uv run agentcli -p "Summarize this repository"
```

Use planning or multi-agent mode:

```bash
uv run agentcli --mode plan -p "Read the README and verify the project structure" --json
uv run agentcli --mode team --worker-mode plan -p "Review the core modules in parallel" --json
```

Inspect the local environment:

```bash
uv run agentcli doctor --cwd .
```

## Configuration

Configuration is loaded in this order:

1. Built-in defaults
2. `~/.agentcli/config.json`
3. Project-level `.agentcli/config.json`
4. Project-level `.env`
5. CLI flags
6. Process environment variables

Use `.env.example` or `.agentcli/config.example.json` as a starting point for local setup.

Example `.env`:

```dotenv
AGENTCLI_PROVIDER=deepseek
AGENTCLI_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=your_key_here
```

You can also use a generic AgentCLI key:

```dotenv
AGENTCLI_PROVIDER=deepseek
AGENTCLI_MODEL=deepseek-v4-flash
AGENTCLI_API_KEY=your_key_here
```

Provider-specific keys currently include:

- `DEEPSEEK_API_KEY`
- `ZAI_API_KEY`
- `GLM_API_KEY`
- `STEP_API_KEY`
- `KIMI_API_KEY`

Temporarily override provider and model:

```bash
uv run agentcli --provider deepseek --model deepseek-v4-flash
```

Connect to a local OpenAI-compatible service:

```bash
AGENTCLI_PROVIDER=openai-compatible \
AGENTCLI_BASE_URL=http://127.0.0.1:11434/v1 \
AGENTCLI_MODEL=qwen2.5-coder \
uv run agentcli -p "Explain this repository"
```

## REPL Commands

```text
/help
/exit
/clear
/context
/memory
/memory search <query>
/memory stats
/memory delete <id>
/memory clear
/save <fact>
/config
/tools
/hitl default|auto
/policy
/audit [N]
/index [path]
/search <query>
/plan <task>
/team <task>
/team --plan <task>
/model
/model <model-id>
/model <provider> <model-id>
/usage
/skill
/skill list
/skill show <name>
/skill on <name>
/skill off <name>
/skill reload
/mcp
/task
/task add [--mode react|plan|team] <task>
/task cancel <task_id>
/task log <task_id>
/snapshot
/snapshot clean
/restore <snapshot-id-or-index>
```

## Built-In Tools

- `read_file`
- `write_file`
- `list_dir`
- `glob` / `glob_files`
- `grep` / `grep_code`
- `bash` / `execute_command`
- `web_search`
- `web_fetch`
- `save_memory`
- `search_memory`
- `load_skill`
- `save_skill`
- `search_code`
- `revert_turn`

File writes, command execution, remote MCP write tools, snapshot restore, and skill persistence are routed through the policy/HITL/audit layer.

## Memory And Context

AgentCLI uses three memory layers:

- Short-term memory: current thread/session messages and tool results
- Static long-term memory: `AGENTS.md`, `AGENTCLI.md`, `.agentcli/AGENTCLI.md`, and configured prompt files
- Dynamic long-term memory: project-scoped SQLite records with kind, source, importance, confidence, TTL, access count, and content hash

The prompt is split into a cache-friendly static prefix and a request-specific dynamic suffix. When the available input budget reaches the configured compression threshold, older conversation turns are summarized while recent messages and complete tool-call pairs are preserved.

## MCP

Configure Chrome DevTools MCP:

```bash
uv run agentcli mcp init-chrome --scope project
```

List configured MCP servers:

```bash
uv run agentcli mcp list
```

Expose AgentCLI tools as an MCP server:

```bash
uv run agentcli mcp serve --transport stdio
uv run agentcli mcp serve --transport http --port 3000
```

## Runtime API

Start the HTTP runtime:

```bash
AGENTCLI_RUNTIME_API_KEY=dev-key \
uv run agentcli serve --http --port 8080
```

Create a thread:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/threads \
  -H 'x-api-key: dev-key'
```

Send a turn:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/threads/<thread_id>/turns \
  -H 'content-type: application/json' \
  -H 'x-api-key: dev-key' \
  -d '{"message":"Summarize this project"}'
```

Create a background task:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/tasks \
  -H 'content-type: application/json' \
  -H 'x-api-key: dev-key' \
  -d '{"message":"Analyze this repository in the background","mode":"plan"}'
```

Run a worker without exposing HTTP:

```bash
uv run agentcli worker --workers 2 --cwd .
```

## SDK

```python
from agentcli.sdk import create_default_engine

engine = create_default_engine(cwd=".")
result = engine.ask_complete("Explain this project")
print(result.text)

plan_result = engine.plan_complete("Read README first, then summarize the architecture")
team_result = engine.team_complete("Ask multiple agents to inspect core modules")
```

## Development

```bash
uv sync --extra dev
uv run python -m ruff check .
uv run python -m ruff format --check .
uv run python -m pytest
uv build
```

Smoke checks:

```bash
uv run agentcli --version
uv run agentcli --help
uv run agentcli doctor --cwd .
uv run agentcli --plain -p hello
```

## License

MIT. See [LICENSE](LICENSE).
