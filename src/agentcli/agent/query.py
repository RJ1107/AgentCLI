from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agentcli.config import AgentCliConfig
from agentcli.context import ContextBudget, ContextWindowManager
from agentcli.image import parse_image_references
from agentcli.llm.base import LlmClient
from agentcli.prompt import PromptAssembler
from agentcli.skill import SkillRegistry
from agentcli.tools.base import ToolContext
from agentcli.tools.executor import ToolExecutor
from agentcli.tools.registry import ToolRegistry
from agentcli.types import Message, Usage


async def query(
    *,
    llm_client: LlmClient,
    tool_registry: ToolRegistry,
    system_prompt: str,
    user_message: str,
    history: list[Message] | None,
    cwd: str,
    config: AgentCliConfig,
    approval_callback=None,
    skill_context_buffer=None,
    max_turns: int = 20,
) -> AsyncIterator[dict[str, Any]]:
    original_user_message = user_message
    user_message = _prepend_skill_candidates(user_message, cwd, config)
    user_message = _prepend_skill_context(user_message, skill_context_buffer)
    messages = [
        *(history or []),
        Message(role="user", content=parse_image_references(user_message, cwd)),
    ]
    tool_definitions = tool_registry.definitions()
    executor = ToolExecutor(tool_registry)
    context = ToolContext(
        cwd=cwd,
        config=config,
        approval_callback=approval_callback,
        skill_context_buffer=skill_context_buffer,
    )
    dynamic_prompt = PromptAssembler(
        config=config,
        cwd=cwd,
        tool_names=tool_registry.list_names(),
        model=llm_client.model_name,
        provider=llm_client.provider_name,
    ).build_dynamic(original_user_message)
    effective_system_prompt = f"{system_prompt}\n\n{dynamic_prompt}".strip()
    context_manager = ContextWindowManager(
        ContextBudget(
            context_window=llm_client.max_context_window,
            max_output_tokens=config.llm.max_tokens,
            compression_threshold=config.memory.compression_threshold,
            compression_target=config.memory.compression_target,
            reserve_tokens=config.memory.compression_reserve_tokens,
        ),
        max_history_messages=config.memory.max_conversation_history,
        min_recent_messages=config.memory.min_recent_messages,
        summary_max_chars=config.memory.summary_max_chars,
    )

    total_usage = Usage()
    turn = 0

    while turn < max_turns:
        turn += 1
        text = ""
        thinking = ""
        stop_reason = "end_turn"
        turn_usage = Usage()
        tool_states: dict[int, dict[str, Any]] = {}

        if config.features.context_compression:
            compression = context_manager.prepare(
                messages,
                system_prompt=effective_system_prompt,
                tool_definitions=tool_definitions,
            )
            messages = compression.messages
            if compression.compressed:
                yield {
                    "type": "context_compressed",
                    "before_tokens": compression.estimated_tokens_before,
                    "after_tokens": compression.estimated_tokens_after,
                    "summarized_messages": compression.summarized_messages,
                }

        async for event in llm_client.chat(
            messages,
            tool_definitions,
            system_prompt=effective_system_prompt,
        ):
            event_type = event.get("type")
            if event_type == "text_delta":
                delta = str(event.get("text") or "")
                text += delta
                yield {"type": "text_delta", "text": delta}
            elif event_type == "thinking_delta":
                delta = str(event.get("thinking") or "")
                thinking += delta
                yield {"type": "thinking_delta", "thinking": delta}
            elif event_type == "tool_call_delta":
                _merge_tool_delta(tool_states, event["tool_call"])
            elif event_type == "message_end":
                stop_reason = str(event.get("stop_reason") or "end_turn")
            elif event_type == "usage":
                usage = Usage.from_mapping(event.get("usage") or {})
                turn_usage = turn_usage + usage
                yield {"type": "usage", "usage": usage.to_dict()}
            elif event_type == "error":
                yield {"type": "error", "error": event["error"]}
                return

        total_usage = total_usage + turn_usage
        tool_calls = _finalize_tool_calls(tool_states)
        assistant_message = Message(role="assistant", content=text, tool_calls=tool_calls)
        if thinking and text:
            assistant_message.content = text
        elif thinking:
            assistant_message.content = ""
        messages.append(assistant_message)
        yield {"type": "turn_complete", "turn": turn, "stop_reason": stop_reason}

        if stop_reason != "tool_use" and not tool_calls:
            break

        for call in tool_calls:
            name = call.get("function", {}).get("name", "unknown")
            yield {"type": "tool_call", "name": name, "input": _tool_input(call)}

        tool_results = await executor.execute_all(tool_calls, context)
        loaded_skill_context = _drain_skill_context(skill_context_buffer)
        load_skill_ids = [
            str(call.get("id") or "")
            for call in tool_calls
            if str(call.get("function", {}).get("name") or "") == "load_skill"
        ]
        injection_target = load_skill_ids[-1] if load_skill_ids else ""
        injected = False
        for result in tool_results:
            yield {
                "type": "tool_result",
                "name": _tool_name_by_id(tool_calls, result.tool_use_id or ""),
                "result": result.content,
                "is_error": result.is_error,
            }
            model_content = result.content
            if loaded_skill_context and result.tool_use_id == injection_target:
                model_content = f"{model_content}\n\n{loaded_skill_context}"
                injected = True
            messages.append(
                Message(
                    role="tool",
                    content=model_content,
                    tool_call_id=result.tool_use_id,
                )
            )
        if loaded_skill_context and not injected:
            messages.append(
                Message(
                    role="user",
                    content=(
                        f"{loaded_skill_context}\n\n"
                        "Use these loaded instructions to continue the current request."
                    ),
                )
            )

    done_event: dict[str, Any] = {
        "type": "done",
        "total_turns": turn,
        "total_tokens": total_usage.total_tokens,
        "usage": total_usage.to_dict(),
        "messages": messages,
    }
    costs = _calculate_costs(llm_client, total_usage)
    if costs:
        done_event["cost"] = costs
    yield done_event


def _merge_tool_delta(tool_states: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    index = int(delta.get("index") or 0)
    state = tool_states.setdefault(
        index,
        {
            "id": delta.get("id") or f"tool_{index}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if delta.get("id"):
        state["id"] = delta["id"]
    function = delta.get("function") or {}
    if function.get("name"):
        state["function"]["name"] = function["name"]
    if function.get("arguments"):
        state["function"]["arguments"] += function["arguments"]


def _finalize_tool_calls(tool_states: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for index in sorted(tool_states):
        state = tool_states[index]
        if state["function"]["name"]:
            calls.append(state)
    return calls


def _tool_input(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("function", {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _tool_name_by_id(calls: list[dict[str, Any]], tool_call_id: str) -> str:
    for call in calls:
        if call.get("id") == tool_call_id:
            return str(call.get("function", {}).get("name") or "unknown")
    return "unknown"


def _prepend_skill_context(user_message: str, skill_context_buffer) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return user_message
    drained = skill_context_buffer.drain()
    if not drained:
        return user_message
    return f"{drained}\n\n---\nUser request:\n{user_message}"


def _drain_skill_context(skill_context_buffer) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return ""
    return skill_context_buffer.drain()


def _prepend_skill_candidates(user_message: str, cwd: str, config: AgentCliConfig) -> str:
    if not config.features.skill:
        return user_message
    candidates = SkillRegistry(cwd).match(user_message, top_k=5)
    if not candidates:
        return user_message
    lines = [
        "Relevant skill candidates for this request:",
        "Call load_skill(name) before proceeding when a candidate applies.",
    ]
    for skill in candidates:
        description = " ".join(skill.description.split())
        if len(description) > 300:
            description = description[:297] + "..."
        tags = f" [tags: {', '.join(skill.tags)}]" if skill.tags else ""
        lines.append(f"- {skill.name}: {description}{tags}")
    candidate_text = "\n".join(lines)
    return f"{candidate_text}\n\n---\nUser request:\n{user_message}"


def _calculate_costs(llm_client: LlmClient, usage: Usage) -> dict[str, Any]:
    calculator = getattr(llm_client, "calculate_cost", None)
    if not callable(calculator):
        return {}
    result: dict[str, Any] = {}
    for currency in ("usd", "cny"):
        try:
            breakdown = calculator(usage, currency=currency)
        except (KeyError, TypeError, ValueError):
            continue
        result[currency] = breakdown.to_dict()
    return result
