from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


@dataclass(frozen=True)
class SentimentConfig:
    """Configuration for sentiment analysis."""
    positive_threshold: float = 0.05
    negative_threshold: float = -0.05


class SentimentAnalyzer:
    """
    Centralized sentiment analyzer using VADER.

    Provides a uniform interface for all text sources (OpenAI, Perplexity, Gemini, Claude, etc.).
    """

    def __init__(self, config: SentimentConfig | None = None) -> None:
        self.config = config or SentimentConfig()
        nltk.download("vader_lexicon", quiet=True)
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Tuple[str, float]:
        """
        Compute sentiment label and score in [-1, 1].

        Returns:
            (label, compound_score), where label is "bullish", "bearish", or "neutral".
        """
        try:
            scores = self._analyzer.polarity_scores(text)
            c = scores["compound"]
            if c >= self.config.positive_threshold:
                return "bullish", c
            if c <= self.config.negative_threshold:
                return "bearish", c
            return "neutral", c
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return "neutral", 0.0
