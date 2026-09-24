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
- MCP server mode (official SDK, stdio and Streamable HTTP) exposing AgentCLI tools to other agents
- Runtime API for threads, turns, event logs, and durable background tasks
- Project-scoped SQLite memory with relevance recall, deduplication, TTL, and capacity control
- Layered context compression: stale tool results cleared first, then an LLM rolling summary with a deterministic fallback
- Workspace snapshot taken right before a request's first write (none for read-only requests), with restore support
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

With one `OPENROUTER_API_KEY`, `/model` offers a checked set of OpenRouter models: GPT-6 Astra, Sol and Luna, Claude Opus 5.5, DeepSeek V4 Flash / V4.1 Flash / V4 Pro 0813, GLM-5.3 and GLM-5.3 Flash, Qwen3.8 Flash, and Gemini 3.8 Flash. Each passed a tool-calling run through AgentCLI before being listed, and each carries OpenRouter's published prices (dated in `llm/pricing.py`), so `/usage` and the idle reminder can show costs. Any other OpenRouter model id works too, without built-in prices.

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
/compact [focus]
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
- `glob`
- `grep`
- `bash`
- `web_search`
- `web_fetch`
- `save_memory`
- `search_memory`
- `load_skill`
- `save_skill`
- `search_code`
- `revert_turn`

File writes, command execution, remote MCP write tools, snapshot restore, and skill persistence are routed through the policy/HITL/audit layer. With the default `auto` policy they need approval; in single-prompt mode there is no one to approve, so pass `--hitl never` only inside a sandbox you are willing to let the agent change.

An MCP tool normally needs approval. Its own `readOnlyHint` lets it skip approval only when its server entry in `.agentcli/mcp.json` sets `"trusted": true`, because a server describes itself and an untrusted one could claim to be read-only.

## Memory And Context

AgentCLI uses three memory layers:

- Short-term memory: current thread/session messages and tool results
- Static long-term memory: `AGENTS.md`, `AGENTCLI.md`, `.agentcli/AGENTCLI.md`, and configured prompt files
- Dynamic long-term memory: project-scoped SQLite records with kind, source, importance, confidence, TTL, access count, and content hash

The system prompt is fully static for a session. Per-request context (date, working directory, recalled memories, skill candidates) is attached to the user message that triggered it and then stays frozen in history, so the provider prefix cache covers the system prompt, tool definitions, and all earlier turns.

When the estimated input reaches `memory.compression_threshold` of the budget, compression runs in layers and stops as soon as the request fits under `memory.compression_target`:

1. Old tool results beyond the newest `memory.keep_recent_tool_results` are replaced with short stubs.
2. Older turns are folded into one rolling summary, written by `memory.summary_model` (empty means the session model) when the older part exceeds `memory.min_llm_summary_tokens`; otherwise, or on failure, an extractive summary is used. Set `memory.llm_summary` to `false` to never call a model.
3. Oversized tool payloads in the retained turns are truncated.

Recent turns and complete tool-call/result pairs are always kept verbatim. The summary is session state and is never written to long-term memory.

`/compact [focus]` summarizes the conversation on demand, whatever its size; the optional focus names what to keep in the most detail (for example `/compact the database migration decisions`).

Providers keep the prompt cache only for a while and do not report when it expires. When you come back after `memory.cache_ttl_minutes` (default 60) with a context of at least `memory.idle_reminder_min_tokens` (default 30,000), the REPL says so before sending: the next request will re-read the whole context at full price, with the extra cost shown when the model has a price table. You can send anyway, compact first, or cancel. After `memory.stale_session_hours` (default 8) it suggests compacting first.

File edits are guarded: `edit_file` requires a unique match (or `replace_all`), and both `edit_file` and `write_file` refuse to change an existing file that has not been read, or that changed since it was last read.

## Routing

Before a request runs in normal mode, AgentCLI estimates how big it is. Built-in rules decide instantly when the signal is clear (length, step words such as "首先/然后/最后", listed items, scope words such as "整个/重构/迁移", fan-out words such as "分别/并行"). Only unclear requests go to a classifier, tried in the order of `routing.classifiers`: Jev (`jev_enabled` plus `TYPESAFE_API_KEY`), then a cheap model (`classifier_model`). Each has a timeout, and a classifier that is disabled, slow, or down is skipped, so the rules' verdict is always the fallback. A request judged large gets a prompt to switch to `/plan` or `/team` (or to stop asking for the session).

`/plan` and `/team` can use a different model per role, as `"provider:model"` in `routing`:

```json
{
  "routing": {
    "planner_model": "openrouter:anthropic/claude-opus-5.5",
    "fast_model": "openrouter:openai/gpt-6-luna",
    "strong_model": "openrouter:openai/gpt-6-sol",
    "top_model": "openrouter:anthropic/claude-opus-5.5"
  }
}
```

The planner writes each step as self-contained instructions and marks it easy or hard. Easy steps run on the fast model, hard ones on the strong model. In `/team` the reviewer is always one tier above the worker, so no model grades its own work. After `attempts_per_tier` rejections (default 2) a step moves up one tier, at most `max_escalations` times (default 1); if it is still rejected it fails with the reviewer's issues in the final report instead of retrying forever. Unset roles use the session model, and a role whose provider key is missing falls back to it too.

## Skills

A skill is a folder with a `SKILL.md` manual (YAML frontmatter with `name` and `description`, then instructions) and, optionally, scripts and reference files the manual points to. The format matches the SKILL.md skills published for other agents, so most of them install unchanged.

Skills are found in three places, later ones overriding earlier ones: built-in, `~/.agentcli/skills/<name>/` (every project), and `<project>/.agentcli/skills/<name>/` (one project). Every enabled skill's name and description is listed in the `load_skill` tool, so the model knows it exists; the full manual is only read when a task needs it, and loading it also tells the model the skill's folder and files so it can run the scripts and read the references.

```bash
uv run agentcli skill add examples/skills/finance-qa          # install for all projects
uv run agentcli skill add path/to/skill --scope project       # this project only
uv run agentcli skill list
```

`examples/skills/` has skills to try:

- `pdf-tools`: text, tables, search, merge, and split for PDF files (`pip install pypdf pdfplumber`)
- `office-docs`: read and create Word, Excel, and PowerPoint files (`pip install python-docx openpyxl python-pptx`)
- `finance-qa`: company and stock questions, with a calculator script so figures are computed, not guessed
- `trump-style`, `sun-yuchen-style`: parody voices for code reviews and release notes

The document skills are original code on open-source libraries; their scripts run on the Python found on your `PATH`.

## MCP

Configure browsers for the agent (pages that need JavaScript, or a login):

```bash
uv run agentcli mcp init-chrome --scope user
```

This writes two `chrome-devtools-mcp` servers, both with a pinned version and usage reporting off:

- `chrome-devtools`: headless, with a throwaway `--isolated` profile, for public pages that only need JavaScript to render.
- `chrome-visible`: a window you can see, with a saved AgentCLI-only profile in `~/.agentcli/browser-profile`, separate from your own Chrome. Use it for sites behind a login or a bot check: you sign in or complete the check yourself in that window, and the login is kept for next time. Delete the folder to forget every login. `--no-visible` skips it.

Both are marked `"trusted": true`, so the server's own read-only tools skip approval while navigation, snapshots, clicks, and scripts still ask.

MCP servers cost nothing until they are used:

- Startup reads each server's tool list from a cache in `~/.agentcli/mcp-cache` instead of launching it. The cache is keyed on the server's command, arguments, and environment, so editing the config refreshes it; `agentcli mcp refresh` clears it.
- MCP tools are deferred: requests carry only their names (in the `load_tools` tool), and the model loads the full definitions of the tools a task needs. Set `"defer": false` on a server to always send its tools.
- A server starts on the first call to one of its tools, keeps one connection for the rest of the session (a page opened by `navigate_page` is still there for `take_snapshot`), and is stopped with the browsers it launched when the session ends.

List configured MCP servers:

```bash
uv run agentcli mcp list
```

Expose AgentCLI tools as an MCP server, so another agent or IDE can use them:

```bash
uv run agentcli mcp serve --transport stdio --cwd /path/to/project
uv run agentcli mcp serve --transport http --port 3000      # http://127.0.0.1:3000/mcp
uv run agentcli mcp serve --read-only                       # only tools that change nothing
```

The server is built on the official MCP SDK. Each tool is annotated as read-only or destructive so the connecting client can ask for approval accordingly; AgentCLI's path and command guards still apply. HTTP listens on 127.0.0.1 only and rejects requests whose Host or Origin is not local, so a web page open in your browser cannot drive it. For example, to let Claude Code use AgentCLI's tools on a project:

```bash
claude mcp add agentcli -- uv run --project /path/to/AgentCLI agentcli mcp serve --cwd /path/to/project
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
