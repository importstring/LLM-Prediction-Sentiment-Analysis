from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelTier = Literal["expensive", "medium", "cheap", "extra_cheap"]


@dataclass(frozen=True)
class ModelTierConfig:
    """
    Standard four-tier model configuration for an LLM provider.
    `default` is the tier where going cheaper starts to
    significantly degrade sentiment insight quality.
    """
    name: str
    expensive: str | None = None
    medium: str | None = None
    cheap: str | None = None
    extra_cheap: str | None = None
    default: str | None = None


OPENAI_MODELS = ModelTierConfig(
    name="openai",
    expensive="gpt-5-pro",
    medium="gpt-4.1",
    cheap="gpt-4o",
    extra_cheap="gpt-4o-mini",
    default="gpt-4o",
)


CLAUDE_MODELS = ModelTierConfig(
    name="claude",
    expensive="claude-opus-4.5",
    medium="claude-sonnet-4.5",
    cheap="claude-haiku-4.5",
    extra_cheap="claude-3.5-haiku",
    default="claude-haiku-4.5",
)


GEMINI_MODELS = ModelTierConfig(
    name="gemini",
    expensive="gemini-3-pro-preview",
    medium="gemini-2.5-pro",
    cheap="gemini-2.5-flash",
    extra_cheap="gemini-2.5-flash-lite",
    default="gemini-2.5-flash",
)


PERPLEXITY_MODELS = ModelTierConfig(
    name="perplexity",
    expensive="sonar-pro",
    medium=None,
    cheap="sonar",
    extra_cheap=None,
    default="sonar",
)


OLLAMA_MODELS = ModelTierConfig(
    name="ollama",
    expensive=None,
    medium=None,
    cheap=None,
    extra_cheap="phi3:mini",
    default="phi3:mini",
)
