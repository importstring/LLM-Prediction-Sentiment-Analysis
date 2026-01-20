from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from anthropic import Anthropic

from algorithm.algopkg.llm_clients.model_tiers import (
    ModelTier,
    model_for,
    max_tokens_for,
    is_supported_model,
    available_models,
)
from algorithm.algopkg.utils.env import get_env


@dataclass(frozen=True)
class ClaudeConfig:
    """
    Configuration for the Claude client.

    Last updated: Jan 2026.
    """

    api_key_env: str = "ANTHROPIC_API_KEY"
    provider_name: str = "claude"
    default_tier: ModelTier | None = "medium"
    default_max_tokens: int = 1_024

    def load_api_key(self) -> str:
        """
        Load the Claude API key from the shared .env setup.

        Raises:
            RuntimeError: If the environment variable is not set or is empty.
        """
        return get_env(self.api_key_env)


class ClaudeClient:
    """
    High-level Claude client based on the official Anthropic SDK.

    Responsibilities:
        - Maintain a configured `Anthropic` client instance.
        - Expose a chat() method with a unified interface for use alongside other LLM clients.
    """

    def __init__(self, config: ClaudeConfig | None = None) -> None:
        """
        Initialize the Claude client.

        Args:
            config: Optional configuration instance. If omitted, a default configuration is used.
        """
        self.config = config or ClaudeConfig()
        api_key = self.config.load_api_key()
        self.client = Anthropic(api_key=api_key)

    def _resolve_model_and_tokens(
        self,
        model: str | None,
        max_tokens: int | None,
    ) -> Tuple[str, int]:
        """
        Decide on model and max_tokens using the shared tier API.
        """
        provider = self.config.provider_name

        # explicit model > tier > provider default
        model_name = model_for(
            provider=provider,
            tier=self.config.default_tier,
            explicit_model=model,
        )

        if not is_supported_model(provider, model_name):
            raise ValueError(
                f"Unsupported model for {provider}: {model_name}. "
                f"Available models: {available_models(provider)}"
            )

        requested = max_tokens if max_tokens is not None else self.config.default_max_tokens
        max_tokens_value = max_tokens_for(provider, model_name, requested)
        return model_name, int(max_tokens_value)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Tuple[str, bool]:
        """
        Execute a chat-style request against the Claude Messages API.

        Args:
            messages: Sequence of message dicts:
                - "role": "user" or "assistant".
                - "content": Text content of the message.
            model: Optional model name override.
            max_tokens: Optional maximum number of output tokens.

        Returns:
            Tuple of (response_text, success_flag).
        """
        if not messages:
            return "No messages provided.", False

        try:
            model_name, max_tokens_value = self._resolve_model_and_tokens(
                model=model,
                max_tokens=max_tokens,
            )
        except ValueError as e:
            return str(e), False

        claude_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        try:
            response = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens_value,
                messages=claude_messages,
            )
            parts: List[str] = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            text = "".join(parts) if parts else str(response.content)
            return text, True
        except Exception as e:
            return f"Error: {e}", False


if __name__ == "__main__":
    client = ClaudeClient()
    text, ok = client.chat(
        [
            {"role": "user", "content": "Explain CAPM in 3 concise bullet points."},
        ]
    )
    if ok:
        print(text)
    else:
        print("Error:", text)
