# AgentCLI Capability Matrix

This document tracks the capabilities implemented in the Python package and the areas that need live credentials or platform state for verification.

## Implemented

- CLI:
  - `agentcli`
  - `agentcli -p`
  - `--provider`
  - `--model`
  - `--plain`
  - `--mode react|plan|team`
  - `--worker-mode react|plan`
  - `--json` usage/cost output
  - `--cwd`
  - `agentcli doctor`
  - `agentcli serve --http --port <port>`
- REPL:
  - `/help`
  - `/clear`
  - `/context`
  - `/memory`
  - `/save`
  - `/config`
  - `/tools`
  - `/hitl`
  - `/policy`
  - `/audit`
  - `/index`
  - `/search`
  - `/plan`
  - `/team`
  - `/model`
  - `/usage`
  - `/task`
  - `/snapshot`
  - `/restore`
  - `/skill`
  - `/mcp`
  - `/exit`
- Agent execution:
  - OpenAI-compatible streaming LLM client
  - ReAct loop with text, thinking, tool-call, tool-result, and usage events
  - Plan-and-Execute DAG planning with dependency-aware batches
  - Multi-Agent orchestration with Planner, Workers, Reviewer, bounded retry, and per-worker `react|plan` mode
  - isolated skill context per sub-agent and parallel plan task
  - SDK entry point for ReAct, Plan, and Team modes
  - pre/post side-history snapshots around agent runs
- Configuration:
  - built-in defaults
  - user config
  - project config
  - project `.env`
  - CLI overrides
  - process env
  - provider-specific keys including `DEEPSEEK_API_KEY`, `GLM_API_KEY`, `STEP_API_KEY`, and `KIMI_API_KEY`
- Tools:
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
- Safety:
  - PathGuard
  - CommandGuard
  - HITL approval
  - JSONL AuditLog
- Memory:
  - static project memory files `AGENTS.md`, `AGENTCLI.md`, `.agentcli/AGENTCLI.md`, and local variants
  - governed SQLite dynamic memory with metadata, deduplication, TTL, quota, access tracking, and relevance recall
  - request-specific Top-K recall plus model-initiated `search_memory`
  - bounded short-term history and deterministic compression
- Skills:
  - built-in, user, and project skill layers
  - user/project `.agentcli/skills/*/SKILL.md`
  - `~/.agentcli/skills.json` disabled-state store
  - `load_skill` with one-shot `SkillContextBuffer` injection
  - name/description/tag Top-K matching with Chinese n-gram support
  - safe project/user `save_skill` with mandatory user approval
- RAG:
  - SQLite local code index
  - `/index`
  - `/search`
  - `search_code`
- MCP:
  - official MCP Python SDK client
  - stdio MCP server connection
  - Streamable HTTP MCP server connection
  - dynamic `mcp__server__tool` registration
  - virtual resource tools
  - virtual prompt tools
  - `agentcli mcp init-chrome`
  - `agentcli mcp list`
  - AgentCLI MCP server over stdio/http for built-in tools
- Runtime:
  - API key requirement
  - `POST /v1/threads`
  - `POST /v1/threads/{id}/turns`
  - `GET /v1/threads/{id}/events`
  - `POST /v1/tasks`
  - `GET /v1/tasks`
  - `GET /v1/tasks/{id}`
  - `POST /v1/tasks/{id}/cancel`
  - SQLite durable task queue
  - task modes `react|plan|team`
  - atomic claim, project scope, lease/heartbeat recovery, and cancellation-safe completion
  - standalone `agentcli worker`
  - persisted Runtime thread history
- Snapshot:
  - `pre-turn` / `post-turn`
  - `/snapshot`
  - `/restore`
  - `revert_turn`
- Image input:
  - `@image:path`
  - `@image:file:///path`
  - `@image:https://...`
  - local image resize/compress
  - transparent PNG white background handling
  - provider/model capability fallback
- Diagnostics:
  - Python syntax diagnostics after `write_file`
- Usage and cost:
  - OpenAI-compatible streaming usage-only chunks
  - input/output/cache-hit/cache-miss/reasoning token aggregation
  - configurable price profiles
  - ReAct/Plan/Team SDK and CLI aggregation

## Live Dependencies

These areas need credentials or external platform state for full live verification:

- Real LLM calls need API keys.
- Chrome DevTools MCP needs Node.js LTS, npm/npx, and Chrome.
- Runtime API turn execution needs a working LLM key.
- Web search depends on public search/fetch availability.

## Verification

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev python -m pytest
uv build
uv run agentcli --help
uv run agentcli doctor --cwd .
uv run agentcli mcp serve --transport http --port 3999
```
