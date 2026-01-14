from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple, Union, List

import aiohttp
import nltk
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class PerplexityConfig:
    """Configuration and supported models for Perplexity usage. Last updated Jan 2026."""

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
    Single-module Perplexity integration.

    Responsibilities:
    - Load and store the API key.
    - Provide synchronous and asynchronous query methods.
    - Handle basic model selection, validation, and sentiment analysis.
    """

    def __init__(self, api_key: str, config: PerplexityConfig | None = None) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("Perplexity API key is empty")

        self.config = config or PerplexityConfig()
        self.default_model = self.config.default_model
        self.default_role = (
            "You are an AI financial analyst providing market insights based on extensive data analysis."
        )

        self._sentiment = self._init_sentiment_analyzer()

    # ---------- public API: sync ----------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(
        self,
        query: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_auto_model: bool = False,
    ) -> Tuple[str, float]:
        """
        Synchronous Perplexity query with optional automatic model selection.

        Returns a tuple (response_text, cost_estimate).
        """
        if not query.strip():
            return "Invalid query provided. Please check the input.", 0.0

        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Perplexity-Version": current_date,
            "Content-Type": "application/json",
        }

        if use_auto_model:
            model_info = self._auto_select_model(query, headers)
            model_id = model_info["id"]
            cost = model_info.get("cost", 0.0)
        else:
            model_id = model or self.default_model
            if not self.config.is_supported(model_id):
                return f"Unsupported model: {model_id}", 0.0
            cost = 0.0

        max_tokens = max_tokens or self.config.default_max_tokens
        max_tokens = self.config.max_tokens_for(model_id, max_tokens)
        temperature = temperature if temperature is not None else self.config.default_temperature

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": self.default_role},
                {"role": "user", "content": query},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
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
            return text, float(cost)
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
        """Asynchronous query against the Perplexity chat completions endpoint."""
        if not query.strip():
            return "Invalid query provided. Please check the input.", False

        model_id = model or self.default_model
        if not self.config.is_supported(model_id):
            return f"Unsupported model: {model_id}", False

        max_tokens = max_tokens or self.config.default_max_tokens
        max_tokens = self.config.max_tokens_for(model_id, max_tokens)
        temperature = temperature if temperature is not None else self.config.default_temperature

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
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
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

    # ---------- sentiment / evaluation ----------

    def evaluate_response(self, response: Union[str, dict]) -> dict:
        """Return a dictionary containing text and sentiment evaluation."""
        try:
            text = response if isinstance(response, str) else response.get("text", "")
            if not text:
                return {"error": "Empty response", "sentiment": "neutral", "sentiment_score": 0.0}

            sentiment, score = self.analyze_sentiment(text)
            return {
                "text": text,
                "sentiment": sentiment,
                "sentiment_score": score,
                "error": None,
            }
        except Exception as e:
            logging.error(f"Evaluation error: {e}")
            return {
                "error": f"Evaluation failed: {e}",
                "sentiment": "neutral",
                "sentiment_score": 0.0,
            }

    def analyze_sentiment(self, statement: str) -> Tuple[str, float]:
        """Compute VADER-based sentiment, mapped to bullish / bearish / neutral."""
        try:
            scores = self._sentiment.polarity_scores(statement)
            c = scores["compound"]
            if c >= 0.05:
                return "bullish", c
            if c <= -0.05:
                return "bearish", c
            return "neutral", c
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return "neutral", 0.0

    # ---------- internals ----------

    def _init_sentiment_analyzer(self) -> SentimentIntensityAnalyzer:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()

    def _auto_select_model(self, query: str, headers: Dict[str, str]) -> dict:
        """
        Perform complexity-based model selection using the /models endpoint.

        This method is used when query(..., use_auto_model=True) is enabled.
        """
        try:
            models_response = requests.get(
                "https://api.perplexity.ai/models",
                headers=headers,
                timeout=10,
            )
            models = models_response.json().get("data", [])
        except Exception as e:
            logging.warning(f"Model discovery failed, falling back to default: {e}")
            return {"id": self.default_model, "cost": 0.0}

        complexity = self._estimate_complexity(query)
        return self._select_model_by_complexity(complexity, models)

    def _estimate_complexity(self, query: str) -> int:
        """
        Simple complexity estimator returning values in the range 1–5.

        Can be replaced later with a more advanced heuristic or local model.
        """
        length = len(query.split())
        if length < 20:
            return 2
        if length < 80:
            return 3
        return 4

    def _select_model_by_complexity(self, complexity: int, models: List[dict]) -> dict:
        """Select a model from the available list based on estimated complexity."""
        if not models:
            return {"id": self.default_model, "cost": 0.0}

        if complexity <= 2:
            return next((m for m in models if "small" in m["id"]), models[0])
        if complexity == 3:
            return next((m for m in models if "medium" in m["id"]), models[0])
        return next((m for m in models if "large" in m["id"]), models[0])


if __name__ == "__main__":
    # Example usage for manual testing.
    key = "YOUR_PPLX_API_KEY_HERE"
    client = PerplexityClient(api_key=key)
    text, cost = client.query(
        "Give a high-level overview of semiconductor ETFs.",
        use_auto_model=True,
    )
    print("Cost estimate:", cost)
    print("Response:", text)
