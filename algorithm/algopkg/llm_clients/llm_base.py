from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Any


class LLMProvider(str, Enum):
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    GEMINI = "gemini"
    CLAUDE = "claude"
    OLLAMA = "ollama"


Message = Dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass(frozen=True)
class ChatResult:
    text: str
    success: bool
    provider: LLMProvider
    model: str | None = None
    meta: Dict[str, Any] | None = None


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.

    Responsibilities:
        - Provide a unified chat() interface across providers.
        - Hide provider-specific SDK details from the Agent.
    """

    provider: LLMProvider

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Execute a chat-style request.

        Args:
            messages: List of messages in {"role", "content"} format.
            model: Optional model name override.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.).

        Returns:
            ChatResult with text, success flag, provider, model, and optional metadata.
        """
        raise NotImplementedError
