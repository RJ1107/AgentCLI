from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from agentcli.types import Usage

PER_MILLION_TOKENS = 1_000_000
DEEPSEEK_PRICING_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"
DEEPSEEK_PRICING_AS_OF = "2026-07-17"
PRICE_CHANGE_NOTICE = (
    "Provider prices can change. Built-in prices are dated defaults; LlmConfig.prices "
    "overrides take precedence."
)


@dataclass(frozen=True, slots=True)
class PerMillionTokenPrices:
    input_cache_hit: float
    input_cache_miss: float
    output: float


@dataclass(frozen=True, slots=True)
class ModelPriceProfile:
    model: str
    context_window: int
    prices: dict[str, PerMillionTokenPrices]
    as_of: str
    source_url: str
    notice: str = PRICE_CHANGE_NOTICE

    def for_currency(self, currency: str) -> PerMillionTokenPrices:
        normalized = currency.lower()
        try:
            return self.prices[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self.prices)) or "none"
            raise ValueError(
                f'Currency "{currency}" is not configured for {self.model}; available: {available}'
            ) from exc


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    currency: str
    input_cache_hit_tokens: int
    input_cache_miss_tokens: int
    input_uncategorized_tokens: int
    output_tokens: int
    reasoning_tokens: int
    input_cache_hit_cost: float
    input_cache_miss_cost: float
    input_uncategorized_cost: float
    output_cost: float
    total_cost: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


DEEPSEEK_V4_PRICE_PROFILES: dict[str, ModelPriceProfile] = {
    "deepseek-v4-flash": ModelPriceProfile(
        model="deepseek-v4-flash",
        context_window=1_000_000,
        prices={
            "usd": PerMillionTokenPrices(
                input_cache_hit=0.0028,
                input_cache_miss=0.14,
                output=0.28,
            ),
            "cny": PerMillionTokenPrices(
                input_cache_hit=0.02,
                input_cache_miss=1.0,
                output=2.0,
            ),
        },
        as_of=DEEPSEEK_PRICING_AS_OF,
        source_url=DEEPSEEK_PRICING_SOURCE,
    ),
    "deepseek-v4-pro": ModelPriceProfile(
        model="deepseek-v4-pro",
        context_window=1_000_000,
        prices={
            "usd": PerMillionTokenPrices(
                input_cache_hit=0.003625,
                input_cache_miss=0.435,
                output=0.87,
            ),
            "cny": PerMillionTokenPrices(
                input_cache_hit=0.025,
                input_cache_miss=3.0,
                output=6.0,
            ),
        },
        as_of=DEEPSEEK_PRICING_AS_OF,
        source_url=DEEPSEEK_PRICING_SOURCE,
    ),
}


def get_builtin_price_profile(model: str) -> ModelPriceProfile | None:
    return DEEPSEEK_V4_PRICE_PROFILES.get(model.lower())


def resolve_price_profile(
    model: str,
    *,
    context_window: int,
    overrides: Mapping[str, Mapping[str, float]] | None = None,
    include_builtin: bool = True,
) -> ModelPriceProfile | None:
    """Resolve dated defaults plus optional config overrides expressed per 1M tokens."""
    builtin = get_builtin_price_profile(model) if include_builtin else None
    if not builtin and not overrides:
        return None

    resolved: dict[str, PerMillionTokenPrices] = dict(builtin.prices) if builtin else {}
    for currency, raw_prices in (overrides or {}).items():
        normalized = currency.lower()
        existing = resolved.get(normalized)
        resolved[normalized] = _merge_currency_prices(normalized, raw_prices, existing)

    return ModelPriceProfile(
        model=model,
        context_window=context_window,
        prices=resolved,
        as_of=builtin.as_of if builtin else "config",
        source_url=builtin.source_url if builtin else "config",
    )


def calculate_cost(
    usage: Usage | Mapping[str, Any],
    profile: ModelPriceProfile,
    *,
    currency: str = "usd",
) -> CostBreakdown:
    """Calculate token costs; unclassified input is conservatively priced as cache miss."""
    normalized_usage = usage if isinstance(usage, Usage) else Usage.from_mapping(dict(usage))
    prices = profile.for_currency(currency)

    input_tokens = normalized_usage.input_tokens
    cache_hit_tokens = min(normalized_usage.cache_hit_tokens, input_tokens)
    remaining = input_tokens - cache_hit_tokens
    cache_miss_tokens = min(normalized_usage.cache_miss_tokens, remaining)
    uncategorized_tokens = remaining - cache_miss_tokens

    cache_hit_cost = _token_cost(cache_hit_tokens, prices.input_cache_hit)
    cache_miss_cost = _token_cost(cache_miss_tokens, prices.input_cache_miss)
    uncategorized_cost = _token_cost(uncategorized_tokens, prices.input_cache_miss)
    output_cost = _token_cost(normalized_usage.output_tokens, prices.output)

    return CostBreakdown(
        currency=currency.lower(),
        input_cache_hit_tokens=cache_hit_tokens,
        input_cache_miss_tokens=cache_miss_tokens,
        input_uncategorized_tokens=uncategorized_tokens,
        output_tokens=normalized_usage.output_tokens,
        reasoning_tokens=normalized_usage.reasoning_tokens,
        input_cache_hit_cost=cache_hit_cost,
        input_cache_miss_cost=cache_miss_cost,
        input_uncategorized_cost=uncategorized_cost,
        output_cost=output_cost,
        total_cost=cache_hit_cost + cache_miss_cost + uncategorized_cost + output_cost,
    )


def _merge_currency_prices(
    currency: str,
    raw: Mapping[str, float],
    existing: PerMillionTokenPrices | None,
) -> PerMillionTokenPrices:
    required = ("input_cache_hit", "input_cache_miss", "output")
    values = asdict(existing) if existing else {}
    values.update({key: float(raw[key]) for key in required if key in raw})
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(
            f"Incomplete {currency} price override; missing per-million fields: "
            + ", ".join(missing)
        )
    if any(values[key] < 0 for key in required):
        raise ValueError(f"Token prices for {currency} must be non-negative")
    return PerMillionTokenPrices(**values)


def _token_cost(tokens: int, per_million_price: float) -> float:
    return tokens * per_million_price / PER_MILLION_TOKENS
