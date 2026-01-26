from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from algopkg.utils.sentiment_analysis import SentimentAnalyzer, SentimentConfig 

# Universal config
ModelTier = Literal["expensive", "medium", "cheap", "extra_cheap"]

class LLMProvider(str, Enum):
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    GEMINI = "gemini"
    CLAUDE = "claude"
    OLLAMA = "ollama"

Message = Dict[str, str]

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
    
    # Define tier-to-model mappings in subclasses
    TIER_MAP: Dict[ModelTier, str] = {}

    def __init__(
        self,
        enable_sentiment: bool = True,
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

    def _resolve_model(
        self,
        model: Optional[str] = None,
        tier: Optional[ModelTier] = None,
    ) -> Optional[str]:
        """
        Resolve model from either explicit model name or tier.
        
        Args:
            model: Explicit model name (e.g., "gpt-4o")
            tier: Model tier (e.g., "expensive", "medium")
            
        Returns:
            Resolved model name, or None for Ollama provider
            
        Raises:
            ValueError: If neither model nor tier is provided (except for Ollama),
                    or if tier is invalid.
        
        Notes:
            - For Ollama provider, always returns None (model parameter is ignored)
            - If both model and tier are provided, tier takes precedence
        """
        # Ollama doesn't need/use model parameter
        if self.provider == LLMProvider.OLLAMA:
            return None
        
        # Tier takes precedence over model if both are provided
        if tier is not None:
            if tier not in self.TIER_MAP:
                raise ValueError(
                    f"Invalid tier '{tier}' for {self.provider.value}. "
                    f"Valid tiers: {list(self.TIER_MAP.keys())}"
                )
            return self.TIER_MAP[tier]
        
        # Use explicit model if provided
        if model is not None:
            return model
        
        # Neither provided (and not Ollama)
        raise ValueError(
            f"Must specify either 'model' or 'tier' for {self.provider.value}."
        )


    @abstractmethod
    def _raw_chat(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Provider-specific chat implementation that returns a ChatResult
        WITHOUT sentiment filled in.
        
        Note: model is now required (non-optional) since it's resolved before calling.
        """
        raise NotImplementedError

    def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        tier: Optional[ModelTier] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Execute a chat-style request and (optionally) attach sentiment.

        Args:
            messages: List of messages in {"role", "content"} format.
            model: Optional explicit model name (e.g., "gpt-4o-mini").
            tier: Optional model tier (e.g., "expensive", "medium").
                  Mutually exclusive with model.
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
                
        Raises:
            ValueError: If both model and tier are specified, or neither is specified.
        """
        # Resolve model from either parameter
        resolved_model = self._resolve_model(model=model, tier=tier)
        
        result = self._raw_chat(messages=messages, model=resolved_model, **kwargs)

        if self._enable_sentiment and self._sentiment_analyzer and result.text:
            label, score = self._sentiment_analyzer.analyze(result.text)

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
