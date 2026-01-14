from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple

import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class PerplexityConfig:
    """
    Configuration and supported models for Perplexity usage.

    Last updated: Jan 2026.
    """
    default_model: str = "llama-3.1-sonar-large-128k-online"
    default_max_tokens: int = 4_096
    default_temperature: float = 0.7

    supported_models: Dict[str, Dict[str, int]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_models",
            {
                "llama-3.1-sonar-large-128k-online": {"max_tokens": 4_096},
                "pplx-7b-online": {"max_tokens": 2_048},
                "pplx-70b-online": {"max_tokens": 4_096},
            },
        )

    def is_supported(self, model: str) -> bool:
        return model in self.supported_models

    def max_tokens_for(self, model: str, requested: int) -> int:
        limit = self.supported_models[model]["max_tokens"]
        return min(requested, limit)


class PerplexityClient:
    """
    High-level Perplexity client.

    Responsibilities:
        - Maintain configuration and API key.
        - Provide synchronous and asynchronous query methods.
    """

    def __init__(self, api_key: str, config: PerplexityConfig | None = None) -> None:
        """
        Initialize the Perplexity client.

        Args:
            api_key: Perplexity API key.
            config: Optional configuration instance.
        """
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("Perplexity API key is empty")

        self.config = config or PerplexityConfig()
        self.default_model = self.config.default_model
        self.default_role = (
            "You are an AI financial analyst providing market insights based on extensive data analysis."
        )

    # ---------- public API: sync ----------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(
        self,
        query: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Tuple[str, float]:
        """
        Execute a synchronous query against the Perplexity chat completions endpoint.

        Args:
            query: User query text.
            model: Optional model name override.
            max_tokens: Optional maximum number of output tokens.
            temperature: Optional sampling temperature.

        Returns:
            Tuple of (response_text, cost_estimate). The cost estimate is currently 0.0.
        """
        if not query.strip():
            return "Invalid query provided. Please check the input.", 0.0

        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Perplexity-Version": current_date,
            "Content-Type": "application/json",
        }

        model_id = model or self.default_model
        if not self.config.is_supported(model_id):
            return f"Unsupported model: {model_id}", 0.0

        max_tokens_value = max_tokens or self.config.default_max_tokens
        max_tokens_value = self.config.max_tokens_for(model_id, max_tokens_value)
        temperature_value = (
            temperature if temperature is not None else self.config.default_temperature
        )

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": self.default_role},
                {"role": "user", "content": query},
            ],
            "temperature": float(temperature_value),
            "max_tokens": int(max_tokens_value),
        }

        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            return text, 0.0
        except Exception as e:
            logging.error(f"Perplexity query failed: {e}")
            return f"Error: {e}", 0.0

    # ---------- public API: async ----------

    async def query_async(
        self,
        session: aiohttp.ClientSession,
        query: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Tuple[str, bool]:
        """
        Execute an asynchronous query against the Perplexity chat completions endpoint.

        Args:
            session: Existing aiohttp client session.
            query: User query text.
            model: Optional model name override.
            max_tokens: Optional maximum number of output tokens.
            temperature: Optional sampling temperature.

        Returns:
            Tuple of (response_text, success_flag).
        """
        if not query.strip():
            return "Invalid query provided. Please check the input.", False

        model_id = model or self.default_model
        if not self.config.is_supported(model_id):
            return f"Unsupported model: {model_id}", False

        max_tokens_value = max_tokens or self.config.default_max_tokens
        max_tokens_value = self.config.max_tokens_for(model_id, max_tokens_value)
        temperature_value = (
            temperature if temperature is not None else self.config.default_temperature
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": self.default_role},
                {"role": "user", "content": query},
            ],
            "temperature": float(temperature_value),
            "max_tokens": int(max_tokens_value),
        }

        try:
            async with session.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    return f"Error: {response.status}", False
                data = await response.json()
                text = data["choices"][0]["message"]["content"]
                return text, True
        except Exception as e:
            logging.error(f"Async Perplexity query failed: {e}")
            return f"Error: {e}", False


if __name__ == "__main__":
    key = "YOUR_PPLX_API_KEY_HERE"
    client = PerplexityClient(api_key=key)
    text, cost = client.query(
        "Give a high-level overview of semiconductor ETFs.",
    )
    print("Cost estimate:", cost)
    print("Response:", text)
