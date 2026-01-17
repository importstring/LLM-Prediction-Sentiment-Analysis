from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from algopkg.utils.sentiment_analysis import SentimentAnalyzer, SentimentConfig 


class LLMProvider(str, Enum):
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    GEMINI = "gemini"
    CLAUDE = "claude"
    OLLAMA = "ollama"


Message = Dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass(frozen=True)
class ChatResult:
    """
    Standardized result for all LLM providers.
    """
    text: str
    success: bool
    provider: LLMProvider
    model: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.

    Responsibilities:
        - Provide a unified chat() interface across providers.
        - Optionally enrich responses with sentiment analysis via SentimentAnalyzer.
        - Hide provider-specific SDK details from the Agent.
    """

    provider: LLMProvider

    def __init__(
        self,
        enable_sentiment: bool = False,
        sentiment_config: Optional[SentimentConfig] = None,
    ) -> None:
        """
        Args:
            enable_sentiment: If True, run sentiment analysis on responses.
            sentiment_config: Optional configuration for sentiment thresholds.
        """
        self._enable_sentiment = enable_sentiment
        self._sentiment_analyzer: Optional[SentimentAnalyzer] = (
            SentimentAnalyzer(sentiment_config) if enable_sentiment else None
        )

    @abstractmethod
    def _raw_chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Provider-specific chat implementation that returns a ChatResult
        WITHOUT sentiment filled in.
        """
        raise NotImplementedError

    def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Execute a chat-style request and (optionally) attach sentiment.

        Args:
            messages: List of messages in {"role", "content"} format.
            model: Optional model name override.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.).

        Returns:
            ChatResult with:
                - text
                - success
                - provider
                - model
                - meta (provider-specific)
                - sentiment (if enabled)
                - sentiment_score (if enabled)
        """
        result = self._raw_chat(messages=messages, model=model, **kwargs)

        if self._enable_sentiment and self._sentiment_analyzer and result.text:
            label, score = self._sentiment_analyzer.analyze(result.text)
            # Create a new ChatResult with sentiment filled in (dataclass is frozen=False, or use replace)
            result = ChatResult(
                text=result.text,
                success=result.success,
                provider=result.provider,
                model=result.model,
                meta=result.meta,
                sentiment=label,
                sentiment_score=score,
            )

        return result
