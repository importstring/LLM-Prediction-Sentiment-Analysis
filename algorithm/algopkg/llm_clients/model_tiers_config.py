from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional


# Public types

ModelTier = Literal["expensive", "medium", "cheap", "extra_cheap"]


# Core configuration object

@dataclass(frozen=True)
class ModelTierConfig:
    """
    Configuration for a single LLM provider.

    - Four price tiers (expensive / medium / cheap / extra_cheap)
      map to concrete model IDs for that provider.
    - `default` is the tier where going cheaper starts to
      significantly degrade quality for your use case.
    - `max_tokens` stores optional per-model output caps.
    """

    name: str

    # Tier → model id
    expensive: str | None = None
    medium: str | None = None
    cheap: str | None = None
    extra_cheap: str | None = None

    # Default model id (usually one of the tier models)
    default: str | None = None

    # Per-model maximum output tokens (optional)
    max_tokens: Dict[str, int] | None = None

    # ----- instance helpers -----

    def all_models(self) -> Iterable[str]:
        """
        Yield all non-None model IDs for this provider.
        Order: expensive, medium, cheap, extra_cheap, default.
        """
        for model_id in (
            self.expensive,
            self.medium,
            self.cheap,
            self.extra_cheap,
            self.default,
        ):
            if model_id is not None:
                yield model_id

    def model_for_tier(self, tier: Optional[ModelTier]) -> str:
        """
        Map a tier (or None) to a concrete model ID.

        If tier is None, use the provider's default model,
        falling back through other tiers if needed.
        """
        if tier is None:
            return (
                self.default
                or self.medium
                or self.expensive
                or self.cheap
                or self.extra_cheap
            )  # type: ignore[return-value]

        if tier == "expensive" and self.expensive:
            return self.expensive
        if tier == "medium" and self.medium:
            return self.medium
        if tier == "cheap" and self.cheap:
            return self.cheap
        if tier == "extra_cheap" and self.extra_cheap:
            return self.extra_cheap

        # Fallback if the requested tier is not defined for this provider
        return (
            self.default
            or self.medium
            or self.expensive
            or self.cheap
            or self.extra_cheap
        )  # type: ignore[return-value]

    def max_tokens_for(self, model: str, requested: int) -> int:
        """
        Clamp `requested` tokens using per-model caps, if defined.

        If no cap exists for this model/provider, return `requested` unchanged.
        """
        if not self.max_tokens:
            return requested
        limit = self.max_tokens.get(model)
        if limit is None:
            return requested
        return min(requested, limit)


# Provider configurations

OPENAI_MODELS = ModelTierConfig(
    name="openai",
    expensive="gpt-5-pro",
    medium="gpt-4.1",
    cheap="gpt-4o",
    extra_cheap="gpt-4o-mini",
    default="gpt-4o",
    max_tokens={
        "gpt-5-pro": 16_000,
        "gpt-4.1": 16_000,
        "gpt-4o": 16_000,
        "gpt-4o-mini": 16_000,
    },
)

CLAUDE_MODELS = ModelTierConfig(
    name="claude",
    expensive="claude-opus-4.5",
    medium="claude-sonnet-4.5",
    cheap="claude-haiku-4.5",
    extra_cheap="claude-3.5-haiku",
    default="claude-haiku-4.5",
    max_tokens={
        "claude-opus-4.5": 8_000,
        "claude-sonnet-4.5": 8_000,
        "claude-haiku-4.5": 8_000,
        "claude-3.5-haiku": 8_000,
    },
)

GEMINI_MODELS = ModelTierConfig(
    name="gemini",
    expensive="gemini-3-pro-preview",
    medium="gemini-2.5-pro",
    cheap="gemini-2.5-flash",
    extra_cheap="gemini-2.5-flash-lite",
    default="gemini-2.5-flash",
    max_tokens={
        "gemini-3-pro-preview": 16_000,
        "gemini-2.5-pro": 16_000,
        "gemini-2.5-flash": 16_000,
        "gemini-2.5-flash-lite": 8_000,
    },
)

PERPLEXITY_MODELS = ModelTierConfig(
    name="perplexity",
    expensive="sonar-pro",
    medium=None,
    cheap="sonar",
    extra_cheap=None,
    default="sonar",
    max_tokens={
        "sonar-pro": 8_000,
        "sonar": 8_000,
    },
)

OLLAMA_MODELS = ModelTierConfig(
    name="ollama",
    expensive=None,
    medium=None,
    cheap=None,
    extra_cheap="phi3:mini",
    default="phi3:mini",
    max_tokens={
        "phi3:mini": 4_000,
    },
)
