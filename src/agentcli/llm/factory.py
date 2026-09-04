from __future__ import annotations

from agentcli.config import LlmConfig
from agentcli.llm.openai_compatible import OpenAICompatibleClient
from agentcli.llm.pricing import resolve_price_profile

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
PROVIDER_BASE_URLS = {
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "kimi": "https://api.moonshot.cn/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "step": "https://api.stepfun.com/v1",
}

MODEL_CONTEXT_WINDOWS = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek-coder": 128_000,
    "glm-5.2": 200_000,
    "glm-5.1": 200_000,
    "glm-4.7": 200_000,
}


def create_llm_client(config: LlmConfig) -> OpenAICompatibleClient:
    provider = config.provider.lower()
    if provider == "deepseek":
        base_url = config.base_url or DEEPSEEK_BASE_URL
        context = config.context_window or MODEL_CONTEXT_WINDOWS.get(config.model.lower(), 64_000)
        return OpenAICompatibleClient(
            provider_name="deepseek",
            model=config.model,
            api_key=config.api_key,
            base_url=base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            max_context_window=context,
            prompt_cache=True,
            price_profile=resolve_price_profile(
                config.model,
                context_window=context,
                overrides=config.prices,
            ),
        )
    if provider in {"openai", "openai-compatible", "compatible"}:
        context = config.context_window or 128_000
        return OpenAICompatibleClient(
            provider_name=provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url or OPENAI_BASE_URL,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            max_context_window=context,
            prompt_cache=False,
            price_profile=resolve_price_profile(
                config.model,
                context_window=context,
                overrides=config.prices,
                include_builtin=False,
            ),
        )
    if provider in PROVIDER_BASE_URLS:
        context = config.context_window or MODEL_CONTEXT_WINDOWS.get(config.model.lower(), 128_000)
        return OpenAICompatibleClient(
            provider_name=provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url or PROVIDER_BASE_URLS[provider],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            max_context_window=context,
            prompt_cache=False,
            price_profile=resolve_price_profile(
                config.model,
                context_window=context,
                overrides=config.prices,
                include_builtin=False,
            ),
        )
    context = config.context_window or 64_000
    return OpenAICompatibleClient(
        provider_name=provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url or DEEPSEEK_BASE_URL,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
        max_context_window=context,
        prompt_cache=False,
        price_profile=resolve_price_profile(
            config.model,
            context_window=context,
            overrides=config.prices,
            include_builtin=False,
        ),
    )
