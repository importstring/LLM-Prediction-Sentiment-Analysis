from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple

import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from algorithm.algopkg.llm_clients.model_tiers import (
    ModelTier,
    model_for,
    max_tokens_for,
    is_supported_model,
    available_models,
)


@dataclass(frozen=True)
class PerplexityConfig:
    """
    Configuration for Perplexity usage.

    Last updated: Jan 2026.
    """

    provider_name: str = "perplexity"
    default_tier: ModelTier | None = None  # use provider default
    default_max_tokens: int = 4_096
    default_temperature: float = 0.7

    default_role: str = (
        "You are an AI financial analyst providing market insights "
        "based on extensive data analysis."
    )


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
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("Perplexity API key is empty")

        self.api_key = api_key
        self.config = config or PerplexityConfig()

    # ---------- internals ----------

    def _headers(self) -> dict:
        """Base headers for Perplexity requests."""
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Perplexity-Version": current_date,
            "Content-Type": "application/json",
        }

    def _resolve_model_and_limits(
        self,
        model: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> tuple[str, int, float]:
        """
        Decide on model, max_tokens, and temperature using the shared tier API.
        """
        provider = self.config.provider_name

        # Use explicit model if provided, otherwise provider default / tier
        chosen_model = model_for(
            provider=provider,
            tier=self.config.default_tier,
            explicit_model=model,
        )

        if not is_supported_model(provider, chosen_model):
            raise ValueError(
                f"Unsupported model for {provider}: {chosen_model}. "
                f"Available models: {available_models(provider)}"
            )

        # Clamp tokens via shared max_tokens_for
        requested_tokens = max_tokens or self.config.default_max_tokens
        clamped_tokens = max_tokens_for(provider, chosen_model, requested_tokens)

        # Temperature with default
        temp = temperature if temperature is not None else self.config.default_temperature

        return chosen_model, clamped_tokens, float(temp)

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

        Returns:
            Tuple of (response_text, cost_estimate). The cost estimate is currently 0.0.
        """
        if not query.strip():
            return "Invalid query provided. Please check the input.", 0.0

        try:
            model_id, max_tokens_value, temperature_value = self._resolve_model_and_limits(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except ValueError as e:
            # Unsupported model or config issue
            return str(e), 0.0

        headers = self._headers()
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": self.config.default_role},
                {"role": "user", "content": query},
            ],
            "temperature": temperature_value,
            "max_tokens": max_tokens_value,
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

        Returns:
            Tuple of (response_text, success_flag).
        """
        if not query.strip():
            return "Invalid query provided. Please check the input.", False

        try:
            model_id, max_tokens_value, temperature_value = self._resolve_model_and_limits(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except ValueError as e:
            return str(e), False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": self.config.default_role},
                {"role": "user", "content": query},
            ],
            "temperature": temperature_value,
            "max_tokens": max_tokens_value,
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


if __name__ == "__main__": # Testing
    key = "YOUR_PPLX_API_KEY_HERE"
    client = PerplexityClient(api_key=key)
    text, cost = client.query(
        "Give a high-level overview of semiconductor ETFs.",
    )
    print("Cost estimate:", cost)
    print("Response:", text)
