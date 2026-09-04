from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        self.input_tokens = _non_negative_int(self.input_tokens)
        self.output_tokens = _non_negative_int(self.output_tokens)
        self.cache_hit_tokens = _non_negative_int(self.cache_hit_tokens)
        self.cache_miss_tokens = _non_negative_int(self.cache_miss_tokens)
        self.reasoning_tokens = _non_negative_int(self.reasoning_tokens)
        self.total_tokens = _non_negative_int(self.total_tokens)
        if not self.input_tokens and (self.cache_hit_tokens or self.cache_miss_tokens):
            self.input_tokens = self.cache_hit_tokens + self.cache_miss_tokens
        if not self.total_tokens:
            self.total_tokens = self.input_tokens + self.output_tokens

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> Usage:
        """Normalize OpenAI/DeepSeek usage fields while preserving local usage field names."""
        value = value or {}
        completion_details = value.get("completion_tokens_details")
        if not isinstance(completion_details, dict):
            completion_details = {}
        return cls(
            input_tokens=_first_int(value, "input_tokens", "prompt_tokens"),
            output_tokens=_first_int(value, "output_tokens", "completion_tokens"),
            cache_hit_tokens=_first_int(
                value,
                "cache_hit_tokens",
                "prompt_cache_hit_tokens",
            ),
            cache_miss_tokens=_first_int(
                value,
                "cache_miss_tokens",
                "prompt_cache_miss_tokens",
            ),
            reasoning_tokens=_first_int(
                value,
                "reasoning_tokens",
                fallback=completion_details.get("reasoning_tokens"),
            ),
            total_tokens=_first_int(value, "total_tokens"),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_hit_tokens=self.cache_hit_tokens + other.cache_hit_tokens,
            cache_miss_tokens=self.cache_miss_tokens + other.cache_miss_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(slots=True)
class QueryResult:
    text: str
    total_tokens: int
    turns: int
    usage: Usage = field(default_factory=Usage)
    cost: dict[str, Any] = field(default_factory=dict)


def _first_int(value: dict[str, Any], *keys: str, fallback: Any = 0) -> int:
    for key in keys:
        if key in value and value[key] is not None:
            return _non_negative_int(value[key])
    return _non_negative_int(fallback)


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
