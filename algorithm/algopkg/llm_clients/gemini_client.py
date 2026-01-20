from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from google import genai
from google.genai import types  # type: ignore[import]

from algorithm.algopkg.llm_clients.model_tiers import (
    ModelTier,
    model_for,
    max_tokens_for,
    is_supported_model,
    available_models,
)


@dataclass(frozen=True)
class GeminiConfig:
    """
    Configuration for the Gemini client.

    Uses shared tier mapping from model_tiers_api.

    Last updated: Jan 2026.
    """

    api_key_env: str = "GEMINI_API_KEY"
    provider_name: str = "gemini"
    default_tier: ModelTier | None = "cheap"
    default_temperature: float = 0.5
    default_max_output_tokens: int = 1_024

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
      - Allow callers to pick cost/quality via tiers.
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

    # ---------- internals ----------

    def _resolve_model_and_limits(
        self,
        model: str | None,
        max_output_tokens: int | None,
        tier: ModelTier | None,
    ) -> Tuple[str, int]:
        """
        Decide on model and max_output_tokens using the shared tier API.
        """
        provider = self.config.provider_name

        chosen_model = model_for(
            provider=provider,
            tier=tier or self.config.default_tier,
            explicit_model=model,
        )

        if not is_supported_model(provider, chosen_model):
            raise ValueError(
                f"Unsupported model for {provider}: {chosen_model}. "
                f"Available models: {available_models(provider)}"
            )

        requested = (
            max_output_tokens
            if max_output_tokens is not None
            else self.config.default_max_output_tokens
        )
        clamped = max_tokens_for(provider, chosen_model, requested)
        return chosen_model, int(clamped)

    # ---------- public API ----------

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str | None = None,
        tier: ModelTier | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> Tuple[str, bool]:
        """
        Execute a chat-style request against the Gemini API.

        Args:
            messages:
                Sequence of message dictionaries with keys:
                  - "role": "user", "assistant", or "system".
                  - "content": Text content of the message.
            model:
                Optional explicit model name override.
                If provided, this wins over the tier.
            tier:
                Cost/quality tier: "expensive", "medium", "cheap", or "extra_cheap"
                (though Gemini config uses only the ones you filled).
            temperature:
                Optional sampling temperature override.
            max_output_tokens:
                Optional maximum number of output tokens.

        Returns:
            Tuple of (response_text, success_flag).
        """
        if not messages:
            return "No messages provided.", False

        try:
            model_name, max_tokens = self._resolve_model_and_limits(
                model=model,
                max_output_tokens=max_output_tokens,
                tier=tier,
            )
        except ValueError as e:
            return str(e), False

        temp = float(
            temperature if temperature is not None else self.config.default_temperature
        )

        contents: List[types.Content] = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            # Gemini expects "user" / "model" roles in the Content API
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
                    temperature=temp,
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
        ],
        tier="cheap",  # or "medium" / "expensive"
    )
    if ok:
        print(text)
    else:
        print("Error:", text)
