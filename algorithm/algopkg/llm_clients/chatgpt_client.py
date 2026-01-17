from __future__ import annotations

import os
from typing import Dict, Tuple

import openai
from openai import OpenAI


class OpenAIConfig:
    """Holds configuration and supported models for OpenAI usage."""

    # Adjusted the models. Last updated Jan 2026
    SUPPORTED_MODELS: Dict[str, Dict[str, int]] = {
        # Flagship general-purpose model
        "gpt-4o": {"max_tokens": 16_000}, # ~16k output, ~128k context
        
        # Newer 4.1 family – great for code / tools
        "gpt-4.1": {"max_tokens": 16_000}, # Up to 1M context

        # Lighter/cheaper variant 
        "gpt-4.1-mini": {"max_tokens": 16_000},
    }

    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_ROLE = (
        "You are a financial analyst with extensive experience in the stock market. "
        "You provide insights based on your current knowledge."
    )

    @classmethod
    def is_supported(cls, model: str) -> bool:
        return model in cls.SUPPORTED_MODELS

    @classmethod
    def max_tokens_for(cls, model: str, requested: int) -> int:
        limit = cls.SUPPORTED_MODELS[model]["max_tokens"]
        return min(requested, limit)


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

        model = model or OpenAIConfig.DEFAULT_MODEL
        if not OpenAIConfig.is_supported(model):
            return (
                f"Unsupported model: {model}. "
                f"Available models: {list(OpenAIConfig.SUPPORTED_MODELS.keys())}",
                False,
            )

        max_tokens = OpenAIConfig.max_tokens_for(model, max_tokens)
        role = role or OpenAIConfig.DEFAULT_ROLE

        try:
            response = self.client.chat.completions.create(
                model=model,
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
    text, ok = client.query("What are the key factors affecting stock market volatility?")
    if ok:
        print("Response:", text)
    else:
        print("Error:", text)
