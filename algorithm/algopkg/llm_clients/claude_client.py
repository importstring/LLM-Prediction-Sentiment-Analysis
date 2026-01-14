from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from anthropic import Anthropic


@dataclass(frozen=True)
class ClaudeConfig:
    """
    Configuration for the Claude client.

    Last updated: Jan 2026.
    """
    api_key_env: str = "ANTHROPIC_API_KEY"
    default_model: str = "claude-4.5-sonnet"
    default_max_tokens: int = 1024

    def load_api_key(self) -> str:
        """
        Load the Claude API key from the configured environment variable.

        Raises:
            RuntimeError: If the environment variable is not set or is empty.
        """
        key = os.getenv(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"Claude API key not found in environment variable {self.api_key_env}"
            )
        return key


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

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Tuple[str, bool]:
        """
        Execute a chat-style request against the Claude Messages API.

        Args:
            messages: Sequence of message dictionaries with keys:
                - "role": "user" or "assistant".
                - "content": Text content of the message.
            model: Optional model name override.
            max_tokens: Optional maximum number of output tokens.

        Returns:
            Tuple of (response_text, success_flag).
        """
        model_name = model or self.config.default_model
        max_tokens_value = int(
            max_tokens if max_tokens is not None else self.config.default_max_tokens
        )

        claude_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        try:
            response = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens_value,
                messages=claude_messages,
            )
            parts = []
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
