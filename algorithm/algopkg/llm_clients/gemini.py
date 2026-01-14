from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from google import genai
from google.genai import types  # type: ignore[import]


@dataclass(frozen=True)
class GeminiConfig:
    """
    Configuration for the Gemini client.

    Last updated: Jan 2026.
    """
    api_key_env: str = "GEMINI_API_KEY"
    default_model: str = "gemini-2.0-flash"
    default_temperature: float = 0.5
    default_max_output_tokens: int = 1024

    def load_api_key(self) -> str:
        """
        Load the Gemini API key from the configured environment variable.

        Raises:
            RuntimeError: If the environment variable is not set or is empty.
        """
        key = os.getenv(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"Gemini API key not found in environment variable {self.api_key_env}"
            )
        return key


class GeminiClient:
    """
    High-level Gemini client based on the Google Gen AI SDK.

    Responsibilities:
        - Maintain a configured `genai.Client` instance.
        - Expose a chat() method with a unified interface for use alongside other LLM clients.
    """

    def __init__(self, config: GeminiConfig | None = None) -> None:
        """
        Initialize the Gemini client.

        Args:
            config: Optional configuration instance. If omitted, a default configuration is used.
        """
        self.config = config or GeminiConfig()
        api_key = self.config.load_api_key()
        self.client = genai.Client(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> Tuple[str, bool]:
        """
        Execute a chat-style request against the Gemini API.

        Args:
            messages: Sequence of message dictionaries with keys:
                - "role": "user", "assistant", or "system".
                - "content": Text content of the message.
            model: Optional model name override.
            temperature: Optional sampling temperature override.
            max_output_tokens: Optional maximum number of output tokens.

        Returns:
            Tuple of (response_text, success_flag).
        """
        model_name = model or self.config.default_model
        temperature = float(
            temperature if temperature is not None else self.config.default_temperature
        )
        max_tokens = int(
            max_output_tokens
            if max_output_tokens is not None
            else self.config.default_max_output_tokens
        )

        contents: List[types.Content] = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "assistant":
                role = "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text)],
                )
            )

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text, True
        except Exception as e:
            return f"Error: {e}", False


if __name__ == "__main__":
    client = GeminiClient()
    text, ok = client.chat(
        [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": "Explain CAPM in 3 concise bullet points."},
        ]
    )
    if ok:
        print(text)
    else:
        print("Error:", text)
