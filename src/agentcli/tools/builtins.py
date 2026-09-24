from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from agentcli.memory import MemoryManager
from agentcli.policy import CommandGuard
from agentcli.rag import CodeIndex
from agentcli.skill import SkillRegistry
from agentcli.snapshot import SnapshotService
from agentcli.tools import file_ops as fops
from agentcli.tools.base import Tool, ToolContext, ToolResult, object_schema
from agentcli.tools.file_ops import FileOpResult
from agentcli.web import fetch_url, search_web

# asyncio's shell is cmd.exe on Windows and /bin/sh elsewhere. Saying so up front keeps the
# model from writing `ls`, `cat`, or `&&`-heavy POSIX commands for a shell that is not there.
_SHELL_DESCRIPTION = "Execute a shell command in the current workspace. " + (
    "The shell is Windows cmd.exe: use dir, type, findstr, and backslash paths, or run "
    "`python -c` / `powershell -Command` for anything more."
    if os.name == "nt"
    else "The shell is /bin/sh."
)


def get_builtin_tools() -> list[Tool]:
    tools = [
        Tool(
            name="read_file",
            description="Read a text file from the current workspace.",
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "Path to read"},
                    "offset": {"type": "number", "description": "Start line, 1-based"},
                    "limit": {"type": "number", "description": "Maximum number of lines"},
                },
                ["path"],
            ),
            required_keys=["path"],
            handler=_read_file,
        ),
        Tool(
            name="write_file",
            description=(
                "Write a UTF-8 text file inside the current workspace. To overwrite an existing "
                "file, read it first; prefer edit_file for partial changes."
            ),
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "Path to write"},
                    "content": {"type": "string", "description": "File content"},
                    "append": {"type": "boolean", "description": "Append instead of overwrite"},
                },
                ["path", "content"],
            ),
            required_keys=["path", "content"],
            handler=_write_file,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="medium",
            requires_approval=True,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace exact text in a file. The file must have been read first. old_text must "
                "match exactly once unless replace_all is true; include surrounding lines to "
                "make it unique. Returns a diff of the change."
            ),
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_text": {
                        "type": "string",
                        "description": "Text to search for — must match exactly",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Text to replace with",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of exactly one",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Preview changes without modifying the file",
                    },
                },
                ["path", "old_text", "new_text"],
            ),
            required_keys=["path", "old_text", "new_text"],
            handler=_edit_file,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="medium",
            requires_approval=True,
        ),
        Tool(
            name="list_dir",
            description="List entries in a directory inside the current workspace.",
            parameters=object_schema(
                {"path": {"type": "string", "description": "Directory path"}},
                ["path"],
            ),
            required_keys=["path"],
            handler=_list_dir,
        ),
        Tool(
            name="glob",
            description="Find files by glob pattern inside the current workspace.",
            parameters=object_schema(
                {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "limit": {"type": "number", "description": "Maximum results"},
                },
                ["pattern"],
            ),
            required_keys=["pattern"],
            handler=_glob_files,
        ),
        Tool(
            name="grep",
            description="Search text in workspace files.",
            parameters=object_schema(
                {
                    "pattern": {"type": "string", "description": "Regex or plain text pattern"},
                    "path": {"type": "string", "description": "Optional path to search"},
                    "regex": {"type": "boolean", "description": "Treat pattern as regex"},
                    "limit": {"type": "number", "description": "Maximum matches"},
                },
                ["pattern"],
            ),
            required_keys=["pattern"],
            handler=_grep,
        ),
        Tool(
            name="directory_tree",
            description=(
                "Get a recursive tree view of files and directories as indented text. "
                "Each entry shows the name and type. Files have no children, "
                "while directories always show their contents."
            ),
            parameters=object_schema(
                {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: workspace root)",
                    },
                    "max_depth": {
                        "type": "number",
                        "description": "Maximum recursion depth (default: 3)",
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Directory names to exclude from the tree",
                    },
                },
            ),
            required_keys=[],
            handler=_directory_tree,
        ),
        Tool(
            name="get_file_info",
            description=(
                "Retrieve detailed metadata about a file or directory — size, "
                "modification time, permissions, and type."
            ),
            parameters=object_schema(
                {"path": {"type": "string", "description": "Path to inspect"}},
                ["path"],
            ),
            required_keys=["path"],
            handler=_get_file_info,
        ),
        Tool(
            name="bash",
            description=_SHELL_DESCRIPTION,
            parameters=object_schema(
                {
                    "command": {"type": "string", "description": "Shell command"},
                    "timeout": {"type": "number", "description": "Timeout seconds"},
                },
                ["command"],
            ),
            required_keys=["command"],
            handler=_bash,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="high",
            requires_approval=True,
        ),
        Tool(
            name="web_search",
            description=(
                "Search the web for current information. Returns titles, URLs, and snippets."
            ),
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "number", "description": "Maximum result count"},
                },
                ["query"],
            ),
            required_keys=["query"],
            handler=_web_search,
        ),
        Tool(
            name="web_fetch",
            description="Fetch a public HTTP/HTTPS page and return readable text.",
            parameters=object_schema(
                {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_length": {"type": "number", "description": "Maximum returned characters"},
                },
                ["url"],
            ),
            required_keys=["url"],
            handler=_web_fetch,
        ),
        Tool(
            name="save_memory",
            description=(
                "Save an explicit or durable fact to long-term project memory. Use only for stable "
                "preferences, project constraints, user corrections, or reusable decisions; never "
                "store secrets, temporary task state, raw logs, or uncertain claims."
            ),
            parameters=object_schema(
                {
                    "content": {"type": "string", "description": "Durable fact to remember"},
                    "kind": {
                        "type": "string",
                        "enum": ["fact", "preference", "constraint", "correction", "decision"],
                    },
                    "importance": {"type": "number", "description": "Score from 0 to 1"},
                    "confidence": {"type": "number", "description": "Score from 0 to 1"},
                    "expires_at": {
                        "type": "string",
                        "description": "Optional ISO-8601 expiry for time-sensitive memory",
                    },
                },
                ["content"],
            ),
            required_keys=["content"],
            handler=_save_memory,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="medium",
        ),
        Tool(
            name="search_memory",
            description=(
                "Search relevance-ranked long-term project memory when prior preferences, "
                "corrections, constraints, or decisions may matter."
            ),
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "What to recall"},
                    "limit": {"type": "number", "description": "Maximum results"},
                    "kinds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional memory kinds",
                    },
                },
                ["query"],
            ),
            required_keys=["query"],
            handler=_search_memory,
        ),
        Tool(
            name="load_skill",
            description="Load a named AgentCLI skill manual from user/project skill directories.",
            parameters=object_schema(
                {"name": {"type": "string", "description": "Skill name"}},
                ["name"],
            ),
            required_keys=["name"],
            handler=_load_skill,
        ),
        Tool(
            name="save_skill",
            description=(
                "Persist a reusable AgentCLI workflow as a project or user skill after "
                "user approval."
            ),
            parameters=object_schema(
                {
                    "name": {"type": "string", "description": "Lowercase skill slug"},
                    "description": {
                        "type": "string",
                        "description": "When this skill should be used",
                    },
                    "content": {"type": "string", "description": "Reusable skill instructions"},
                    "scope": {
                        "type": "string",
                        "enum": ["project", "user"],
                        "description": "Where to persist the skill",
                    },
                    "version": {"type": "string", "description": "Skill version"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Matching keywords",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Update an existing skill instead of failing",
                    },
                },
                ["name", "description", "content"],
            ),
            required_keys=["name", "description", "content"],
            handler=_save_skill,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="medium",
            requires_approval=True,
        ),
        Tool(
            name="search_code",
            description="Search the local code index for semantically relevant lines.",
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "number", "description": "Maximum matches"},
                },
                ["query"],
            ),
            required_keys=["query"],
            handler=_search_code,
        ),
        Tool(
            name="revert_turn",
            description="Restore the workspace to a previous AgentCLI side-history snapshot.",
            parameters=object_schema(
                {"snapshot": {"type": "string", "description": "Snapshot id or 1-based index"}},
                ["snapshot"],
            ),
            required_keys=["snapshot"],
            handler=_revert_turn,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="high",
            requires_approval=True,
        ),
    ]
    return tools


# ---------------------------------------------------------------------------
# Handler: file operations (thin wrappers over file_ops)
# ---------------------------------------------------------------------------


async def _read_file(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    in_skill = _in_skill_folder(context, str(payload["path"]))
    result: FileOpResult = fops.read_file(
        context.cwd,
        str(payload["path"]),
        offset=int(payload.get("offset") or 1),
        limit=int(payload.get("limit") or 500),
        path_guard_enabled=context.config.policy.path_guard_enabled and not in_skill,
    )
    if not result.is_error and not in_skill:
        _remember_file(context, str(payload["path"]))
    return _to_tool_result(result)


async def _write_file(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    append = bool(payload.get("append"))
    if not append:
        stale = _stale_file_error(context, str(payload["path"]))
        if stale:
            return ToolResult(stale, is_error=True)
    result: FileOpResult = fops.write_file(
        context.cwd,
        str(payload["path"]),
        str(payload["content"]),
        append=append,
        path_guard_enabled=context.config.policy.path_guard_enabled,
    )
    if not result.is_error:
        _remember_file(context, str(payload["path"]))
    return _to_tool_result(result)


async def _edit_file(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    dry_run = bool(payload.get("dry_run"))
    if not dry_run:
        stale = _stale_file_error(context, str(payload["path"]))
        if stale:
            return ToolResult(stale, is_error=True)
    result: FileOpResult = fops.edit_file(
        context.cwd,
        str(payload["path"]),
        str(payload["old_text"]),
        str(payload["new_text"]),
        path_guard_enabled=context.config.policy.path_guard_enabled,
        dry_run=dry_run,
        replace_all=bool(payload.get("replace_all")),
    )
    if not result.is_error and not dry_run:
        _remember_file(context, str(payload["path"]))
    return _to_tool_result(result)


def _file_key(context: ToolContext, path: str) -> tuple[str, int | None]:
    resolved = fops.resolve_path(context.cwd, path, context.config.policy.path_guard_enabled)
    try:
        mtime = resolved.stat().st_mtime_ns if resolved.is_file() else None
    except OSError:
        mtime = None
    return str(resolved), mtime


def _remember_file(context: ToolContext, path: str) -> None:
    key, mtime = _file_key(context, path)
    if mtime is not None:
        context.file_state[key] = mtime


def _stale_file_error(context: ToolContext, path: str) -> str:
    """Refuse to change a file whose current content this agent has not seen.

    Without this, a write based on an old read silently discards whatever changed in between:
    the user's edit in their editor, a formatter run through bash, or a parallel worker.
    """

    key, mtime = _file_key(context, path)
    if mtime is None:
        # New file: nothing to lose. A missing file for edit_file is reported by the edit.
        return ""
    seen = context.file_state.get(key)
    if seen is None:
        return (
            f"{path} has not been read in this session. Read it before changing it, so the "
            "change is based on its current content."
        )
    if seen != mtime:
        return (
            f"{path} was modified after you last read it (by the user, a command, or another "
            "agent). Read it again and redo the change against the current content."
        )
    return ""


async def _list_dir(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    in_skill = _in_skill_folder(context, str(payload["path"]))
    result: FileOpResult = fops.list_directory(
        context.cwd,
        str(payload["path"]),
        path_guard_enabled=context.config.policy.path_guard_enabled and not in_skill,
    )
    return _to_tool_result(result)


async def _glob_files(payload: dict[str, Any], _context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.glob_files(
        _context.cwd,
        str(payload["pattern"]),
        limit=int(payload.get("limit") or 100),
    )
    return _to_tool_result(result)


async def _grep(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.grep(
        context.cwd,
        str(payload["pattern"]),
        path=str(payload.get("path") or "."),
        limit=int(payload.get("limit") or 100),
        use_regex=bool(payload.get("regex", True)),
        path_guard_enabled=context.config.policy.path_guard_enabled,
    )
    return _to_tool_result(result)


async def _directory_tree(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.directory_tree(
        context.cwd,
        str(payload.get("path", ".")),
        max_depth=int(payload.get("max_depth") or 3),
        path_guard_enabled=context.config.policy.path_guard_enabled,
        exclude_patterns=tuple(payload.get("exclude_patterns") or ()),
    )
    return _to_tool_result(result)


async def _get_file_info(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.get_file_info(
        context.cwd,
        str(payload["path"]),
        path_guard_enabled=context.config.policy.path_guard_enabled,
    )
    return _to_tool_result(result)


# ---------------------------------------------------------------------------
# Handler: bash
# ---------------------------------------------------------------------------


async def _bash(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    command = str(payload["command"])
    if context.config.policy.command_guard_enabled:
        CommandGuard(context.config.policy.command_blacklist).validate(command)
    timeout = float(payload.get("timeout") or context.config.tools.timeout)
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=context.cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return ToolResult(f"Command timed out after {timeout:.0f}s", is_error=True)
    output = (stdout + stderr).decode("utf-8", errors="replace")
    if len(output) > 20_000:
        output = output[:20_000] + "\n... [truncated]"
    return ToolResult(
        output or f"(exit {proc.returncode}, no output)",
        is_error=proc.returncode != 0,
    )


# ---------------------------------------------------------------------------
# Handler: web
# ---------------------------------------------------------------------------


async def _web_search(payload: dict[str, Any], _context: ToolContext) -> ToolResult:
    max_results = int(payload.get("max_results") or payload.get("maxResults") or 5)
    try:
        results = await search_web(str(payload["query"]), max_results=max_results)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(f"Search error: {exc}", is_error=True)
    if not results:
        return ToolResult(f'No search results found for "{payload["query"]}".')
    content = "\n\n".join(
        f"{index}. {result.title}\n{result.url}\n{result.snippet}"
        for index, result in enumerate(results, start=1)
    )
    return ToolResult(content, display_summary=f"Search: {len(results)} results")


async def _web_fetch(payload: dict[str, Any], _context: ToolContext) -> ToolResult:
    max_length = int(payload.get("max_length") or payload.get("maxLength") or 10_000)
    try:
        content = await fetch_url(str(payload["url"]), max_length=max_length)
    except Exception as exc:  # noqa: BLE001
        # Some httpx errors (timeouts in particular) stringify to "", which would leave the
        # model guessing. The type name alone already says what went wrong.
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        return ToolResult(f"Fetch error: {detail}", is_error=True)
    return ToolResult(content, display_summary=f"Fetched {payload['url']}")


# ---------------------------------------------------------------------------
# Handler: memory
# ---------------------------------------------------------------------------


async def _save_memory(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    if not context.config.features.memory or not context.config.memory.long_term_enabled:
        return ToolResult("Long-term memory is disabled.", is_error=True)
    manager = MemoryManager(
        context.config.memory.long_term_db_path,
        scope=context.cwd,
        max_entries=context.config.memory.max_long_term_entries,
        max_content_length=context.config.memory.max_memory_chars,
    )
    memory_id = manager.save(
        str(payload["content"]),
        kind=str(payload.get("kind") or "fact"),
        source="agent",
        importance=float(payload.get("importance", 0.5)),
        confidence=float(payload.get("confidence", 1.0)),
        expires_at=payload.get("expires_at"),
    )
    return ToolResult(f"Saved memory #{memory_id}")


async def _search_memory(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    if not context.config.features.memory or not context.config.memory.long_term_enabled:
        return ToolResult("Long-term memory is disabled.", is_error=True)
    raw_kinds = payload.get("kinds")
    if raw_kinds is not None and not isinstance(raw_kinds, list):
        return ToolResult("search_memory kinds must be an array of strings.", is_error=True)
    manager = MemoryManager(
        context.config.memory.long_term_db_path,
        scope=context.cwd,
        max_entries=context.config.memory.max_long_term_entries,
        max_content_length=context.config.memory.max_memory_chars,
    )
    rows = manager.recall(
        str(payload["query"]),
        limit=int(payload.get("limit") or context.config.memory.recall_limit),
        kinds=[str(kind) for kind in raw_kinds] if raw_kinds else None,
        min_score=context.config.memory.recall_min_score,
    )
    if not rows:
        return ToolResult("(no relevant long-term memory)")
    content = "\n".join(
        f"#{row.id} [{row.kind}, importance={row.importance:.2f}] {row.content}" for row in rows
    )
    return ToolResult(content, display_summary=f"Recalled {len(rows)} memories")


# Keep the original handler imports working for SDK users and existing tests.
save_memory = _save_memory
search_memory = _search_memory


# ---------------------------------------------------------------------------
# Handler: skill
# ---------------------------------------------------------------------------


async def _load_skill(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    if not context.config.features.skill:
        return ToolResult("Skills are disabled.", is_error=True)
    skill = SkillRegistry(context.cwd).load(str(payload["name"]))
    if not skill:
        return ToolResult(f'Skill "{payload["name"]}" not found or disabled.', is_error=True)
    content = skill.body or skill.content
    if len(content) > 20_000:
        content = content[:20_000] + "\n... [truncated; use /skill show for the full skill]"
    content += _skill_files_note(skill.path.parent)
    if context.skill_context_buffer:
        context.skill_context_buffer.push(skill.name, content)
        return ToolResult(
            f'Loaded skill "{skill.name}" instructions for the next model turn.',
            display_summary=f"Loaded skill {skill.name}",
        )
    return ToolResult(content, display_summary=f"Loaded skill {skill.name}")


def _skill_files_note(skill_dir) -> str:
    """Tell the model where the skill's other files are.

    A skill is a folder: SKILL.md is the manual, and it may point to scripts to run and
    references to read only when needed. Without the folder path those files are unusable.
    """

    files = sorted(
        path.relative_to(skill_dir).as_posix()
        for path in skill_dir.rglob("*")
        if path.is_file() and path.name != "SKILL.md" and "__pycache__" not in path.parts
    )
    if not files:
        return ""
    listed = "\n".join(f"- {name}" for name in files[:50])
    more = f"\n- ... and {len(files) - 50} more" if len(files) > 50 else ""
    return (
        f"\n\n---\nSkill folder: {skill_dir}\nFiles (read with read_file using the full path; "
        f"run scripts with bash from any directory):\n{listed}{more}"
    )


def _in_skill_folder(context: ToolContext, value: str) -> bool:
    """Skill folders may live outside the workspace (~/.agentcli/skills); reading them is fine."""

    registry = SkillRegistry(context.cwd)
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(context.cwd) / candidate
    resolved = candidate.resolve()
    for root in (registry.builtin_root, registry.user_root, registry.project_skill_root):
        try:
            resolved.relative_to(Path(root).expanduser().resolve())
            return True
        except ValueError:
            continue
    return False


async def _save_skill(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    if not context.config.features.skill:
        return ToolResult("Skills are disabled.", is_error=True)
    raw_tags = payload.get("tags") or []
    if not isinstance(raw_tags, list):
        return ToolResult("save_skill tags must be an array of strings.", is_error=True)
    tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    registry = SkillRegistry(context.cwd)
    scope = str(payload.get("scope") or "project")
    name = str(payload["name"])
    try:
        if bool(payload.get("overwrite")):
            skill = registry.update(
                name,
                description=str(payload["description"]),
                body=str(payload["content"]),
                scope=scope,
                version=str(payload.get("version") or "1.0.0"),
                tags=tags,
            )
        else:
            skill = registry.create(
                name,
                description=str(payload["description"]),
                body=str(payload["content"]),
                scope=scope,
                version=str(payload.get("version") or "1.0.0"),
                tags=tags,
            )
    except (OSError, ValueError) as exc:
        return ToolResult(f"save_skill failed: {exc}", is_error=True)
    return ToolResult(
        f'Saved skill "{skill.name}" in {scope} scope at {skill.path}.',
        display_summary=f"Saved skill {skill.name}",
    )


load_skill = _load_skill


# ---------------------------------------------------------------------------
# Handler: code search
# ---------------------------------------------------------------------------


async def _search_code(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    index = CodeIndex(context.cwd)
    results = index.search(str(payload["query"]), limit=int(payload.get("limit") or 20))
    if not results:
        return ToolResult("(no indexed matches; run /index first)")
    return ToolResult("\n".join(f"{item.path}:{item.line}: {item.snippet}" for item in results))


# ---------------------------------------------------------------------------
# Handler: snapshot revert
# ---------------------------------------------------------------------------


async def _revert_turn(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    record = SnapshotService(context.cwd).restore(str(payload["snapshot"]))
    return ToolResult(f"Restored snapshot {record.id}")


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------


def _to_tool_result(result: FileOpResult) -> ToolResult:
    """Convert a FileOpResult to a ToolResult."""
    return ToolResult(
        content=result.content,
        is_error=result.is_error,
        display_summary=result.display_summary,
    )
