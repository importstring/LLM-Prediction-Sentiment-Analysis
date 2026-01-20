from __future__ import annotations

import os
from typing import Tuple

import openai
from openai import OpenAI

from algorithm.algopkg.llm_clients.model_tiers import (
    ModelTier,
    model_for,
    max_tokens_for,
    is_supported_model,
    available_models,
)


class OpenAIConfig:
    """
    Holds configuration and supported models for OpenAI usage.
    """

    PROVIDER_NAME = "openai"

    DEFAULT_ROLE = (
        "You are a financial analyst with extensive experience in the stock market. "
        "You provide insights based on your current knowledge."
    )


class OpenAIClient:
    """
    Single, high-level interface for all OpenAI calls in this project.

    Responsibilities:
    - Load and hold the API key.
    - Hold a configured `OpenAI` instance.
    - Provide a clean `query` method with sensible defaults.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or self._load_api_key()
        self.client = OpenAI(api_key=self.api_key)

    # ---------- public API ----------

    def query(
        self,
        query: str,
        model: str | None = None,
        tier: ModelTier | None = None,
        max_tokens: int = 4_000,
        temperature: float = 0.5,
        role: str | None = None,
    ) -> Tuple[str, bool]:
        """
        Query OpenAI with standardised error handling.

        Returns (output, success_flag).
        """
        if not query.strip():
            return "Invalid query provided. Please check the input.", False

        provider = OpenAIConfig.PROVIDER_NAME

        # Priority: explicit model > tier > default (handled by model_for)
        chosen_model = model_for(
            provider=provider,
            tier=tier,
            explicit_model=model,
        )

        if not is_supported_model(provider, chosen_model):
            return (
                f"Unsupported model for {provider}: {chosen_model}. "
                f"Available models: {available_models(provider)}",
                False,
            )

        max_tokens = max_tokens_for(provider, chosen_model, max_tokens)
        role = role or OpenAIConfig.DEFAULT_ROLE

        try:
            response = self.client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": query},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stream=False,
                presence_penalty=0,
                frequency_penalty=0,
            )
            output = response.choices[0].message.content
            return output, True

        except openai.BadRequestError as e:
            return f"Invalid request: {e}", False
        except openai.AuthenticationError:
            return "Authentication failed. Check your API key.", False
        except openai.RateLimitError:
            return "Rate limit exceeded. Please try again later.", False
        except openai.APITimeoutError:
            return "Request timed out. Please try again.", False
        except openai.APIConnectionError:
            return "Connection error. Please check your internet connection.", False
        except Exception as e:
            return f"An unexpected error occurred: {e}", False

    # ---------- internals ----------

    def _load_api_key(self) -> str:
        """Centralised key loading logic."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            return api_key

        from os.path import dirname, join

        filepath = join(dirname(__file__), "API_Keys", "OpenAI.txt")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        except Exception:
            raise RuntimeError(
                "OpenAI API key not found in environment variables or API_Keys/OpenAI.txt"
            )

        if not api_key:
            raise RuntimeError("OpenAI API key file is empty")
        return api_key


if __name__ == "__main__":
    client = OpenAIClient()
    text, ok = client.query(
        "What are the key factors affecting stock market volatility?",
        tier="medium",  # or "expensive" / "cheap" / "extra_cheap"
    )
    if ok:
        print("Response:", text)
    else:
        print("Error:", text)
