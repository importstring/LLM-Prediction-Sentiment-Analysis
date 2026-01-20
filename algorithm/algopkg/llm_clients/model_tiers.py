from __future__ import annotations

from typing import Dict, Optional

from algopkg.llm_clients.model_tiers_config import (
    ModelTier,
    ModelTierConfig,
    OPENAI_MODELS,
    CLAUDE_MODELS,
    GEMINI_MODELS,
    PERPLEXITY_MODELS,
    OLLAMA_MODELS,
)


# Registry of providers → config

_PROVIDER_REGISTRY: Dict[str, ModelTierConfig] = {
    "openai": OPENAI_MODELS,
    "claude": CLAUDE_MODELS,
    "gemini": GEMINI_MODELS,
    "perplexity": PERPLEXITY_MODELS,
    "ollama": OLLAMA_MODELS,
}


def get_provider_config(provider: str) -> ModelTierConfig:
    """
    Return the ModelTierConfig for a given provider name.
    """
    return _PROVIDER_REGISTRY[provider]


def is_supported_model(provider: str, model: str) -> bool:
    """
    Check if a model ID is known for this provider.
    """
    cfg = get_provider_config(provider)
    return model in set(cfg.all_models())


def available_models(provider: str) -> list[str]:
    """
    List all known model IDs for this provider, in a stable, de-duplicated order.
    """
    cfg = get_provider_config(provider)
    return list(dict.fromkeys(cfg.all_models()))


def model_for(
    provider: str,
    tier: Optional[ModelTier] = None,
    explicit_model: Optional[str] = None,
) -> str:
    """
    Decide which model ID to use for a given provider.

    Priority:
    1. explicit_model, if provided
    2. model mapped from `tier`
    3. provider default (with safe fallbacks)
    """
    cfg = get_provider_config(provider)
    if explicit_model is not None:
        return explicit_model
    return cfg.model_for_tier(tier)


def max_tokens_for(provider: str, model: str, requested: int) -> int:
    """
    Clamp requested tokens using the provider's per-model caps.
    """
    cfg = get_provider_config(provider)
    return cfg.max_tokens_for(model, requested)
