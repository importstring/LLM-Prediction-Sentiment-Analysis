"""
TickerSelector: Phase 1 of the AI Portfolio Agent

Responsible for:
1. Querying Gemini (cheapest LLM) on all 500 tickers
2. Filtering by confidence threshold
3. Ranking by |sentiment_score|
4. Returning top K candidates for expensive deep-dive research

Cost: ~$0.75 per run (0.5K tokens × 500 tickers × $0.003/1K)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from algopkg.agents.prompt_manager import PromptManager
# TODO: Create a stock data fetcher function



# ============================================================================
# DATA MODELS (Pydantic for type safety)
# ============================================================================

class RawTickerSentiment(BaseModel):
    """Single LLM's sentiment output for one ticker"""
    ticker: str
    sentiment_label: str = Field(description="bullish, neutral, or bearish")
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="-1 (bearish) to +1 (bullish)")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1, how confident in this sentiment")
    rationale: str = Field(default="", description="Why the model thinks this")
    
    @field_validator('sentiment_label')
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in ['bullish', 'neutral', 'bearish']:
            raise ValueError(f"Invalid label: {v}")
        return v


class FilteredTicker(BaseModel):
    """Ticker that passed the confidence filter"""
    ticker: str
    sentiment_label: str
    sentiment_score: float
    confidence: float
    rank: int  # Position after sorting by |sentiment_score|
    absolute_score: float  # |sentiment_score| used for ranking


class SeleccionResult(BaseModel):
    """Final result from TickerSelector.select()"""
    selected_tickers: List[str]
    filtered_tickers: List[FilteredTicker]
    total_scanned: int
    total_passed_filter: int
    filter_threshold: float
    top_bullish: List[str]  # Top 5 bullish tickers
    top_bearish: List[str]  # Top 5 bearish tickers
    timestamp: str
    gemini_cost_cents: float


# ============================================================================
# TICKER SELECTOR
# ============================================================================

class TickerSelector:
    """
    Stage 1: Cheap global scan using Gemini.
    
    Pipeline:
      1. Batch query Gemini on all 500 tickers
      2. Parse + validate responses (Pydantic)
      3. Filter by confidence >= threshold
      4. Rank by |sentiment_score|
      5. Return top K
    """
    
    def __init__(
        self,
        stock_data: Any,  # algopkg.data_models.stock_data.StockData TODO
        llms: Any,  # algopkg.llm_clients.llm_base.BaseLLMClient TODO [adjust auto model selection logic]
        logger: Any,  # algopkg.agents.research_logger.ResearchLogger [TODO]
        confidence_threshold: float = 0.6,
        top_k: int = 120,
        batch_size: int = 25,  # Query in batches to avoid huge context windows
    ):
        """
        Args:
            stock_data: Data provider for ticker universe
            llms: LLM coordinator (BaseLLMClient)
            prompt_manager: Prompt template manager
            logger: Research logger
            confidence_threshold: Minimum confidence to pass filter (0.0-1.0)
            top_k: Number of tickers to select for Phase 2
            batch_size: How many tickers to query in one LLM call
        """
        self.stock_data = stock_data
        self.llms = llms
        self.prompt_manager = PromptManager()
        self.logger = logger
        
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.batch_size = batch_size
        
        self._logger = logging.getLogger(self.__class__.__name__)
    
    
    def select(self, tickers: List[str]) -> List[str]:
        """
        Run the full selection pipeline.
        
        Returns:
            List[str]: Top K tickers selected for expensive deep-dive research
        """
        self._logger.info(f"TickerSelector: Starting scan of {len(tickers)} tickers")
        
        start_time = time.time()
        
        # Stage 1: Scan all tickers
        raw_sentiments = self._scan_all_tickers(tickers)
        
        # Stage 2: Filter + rank
        filtered = self._filter_and_rank(raw_sentiments)
        
        # Stage 3: Select top K
        selected = self._select_top_k(filtered)
        
        elapsed = time.time() - start_time
        self._logger.info(
            f"TickerSelector: Selected {len(selected)} tickers in {elapsed:.1f}s. "
            f"Cost: ${self._estimate_cost(len(tickers)) / 100:.2f}"
        )
        
        return selected
    
    
    def _scan_all_tickers(self, tickers: List[str]) -> List[RawTickerSentiment]:
        """
        Query Gemini in batches on all tickers.
        
        Returns:
            List[RawTickerSentiment]: Parsed sentiment for each ticker
        """
        all_sentiments = []
        
        # Batch tickers to avoid context window explosion
        batches = self._create_batches(tickers, self.batch_size)
        
        self._logger.info(f"Scanning {len(batches)} batches of {self.batch_size} tickers")
        
        for batch_idx, batch in enumerate(batches):
            batch_sentiments = self._query_batch(batch)
            all_sentiments.extend(batch_sentiments)
            
            # Progress logging
            progress_pct = (batch_idx + 1) / len(batches) * 100
            self._logger.info(
                f"  Batch {batch_idx + 1}/{len(batches)} complete "
                f"({progress_pct:.0f}% done, {len(all_sentiments)} parsed)"
            )
        
        return all_sentiments
    
    
    def _query_batch(self, batch: List[str]) -> List[RawTickerSentiment]:
        """
        Query Gemini for sentiment on a batch of tickers.
        
        Args:
            batch: List of tickers (e.g., 25 tickers)
        
        Returns:
            List[RawTickerSentiment]: Parsed responses
        """

        prompt = self.prompt_manager.ticker_sentiment_batch(batch)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial sentiment analyst. "
                    "For each ticker, output ONLY valid JSON (no markdown, no extra text). "
                    "Output a JSON array of objects with keys: "
                    "ticker, sentiment_label (bullish/neutral/bearish), "
                    "sentiment_score (-1 to 1), confidence (0 to 1), rationale."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        
        # Query Gemini (cheapest)
        result = self.llms.chat( # TODO: Adjust based on what LLM Coor needs
            messages=messages,
            provider="gemini",  # TODO: Update if LLMCoordinator API differs
        )
        
        if not result.success:
            self._logger.error(f"Gemini query failed: {result.text}")
            # Graceful fallback: return neutral sentiment for all tickers in batch
            return [
                RawTickerSentiment(
                    ticker=ticker,
                    sentiment_label="neutral",
                    sentiment_score=0.0,
                    confidence=0.0,
                    rationale="(Query failed, defaulting to neutral)",
                )
                for ticker in batch
            ]
        
        # Parse JSON response
        sentiments = self._parse_batch_response(result.text, batch)
        return sentiments
    
    
    def _parse_batch_response(self, response_text: str, expected_tickers: List[str]) -> List[RawTickerSentiment]: # TODO Implement stock data fetcher for this
        """
        Parse LLM's JSON batch response into RawTickerSentiment objects.
        
        LLM should return JSON array like:
        [
          {"ticker": "AAPL", "sentiment_label": "bullish", "sentiment_score": 0.75, ...},
          {"ticker": "MSFT", "sentiment_label": "neutral", "sentiment_score": 0.1, ...},
          ...
        ]
        
        Args:
            response_text: Raw LLM response
            expected_tickers: List of tickers we asked about (for fallback)
        
        Returns:
            List[RawTickerSentiment]: Validated sentiments
        """
        sentiments = []
        
        try:
            # Try to extract JSON array
            data = json.loads(response_text)
            
            if not isinstance(data, list):
                # Maybe it's wrapped in an object?
                if isinstance(data, dict) and "tickers" in data:
                    data = data["tickers"]
                elif isinstance(data, dict) and "sentiments" in data:
                    data = data["sentiments"]
                else:
                    raise ValueError(f"Expected JSON array, got: {type(data)}")
            
            # Parse each entry
            for entry in data:
                try:
                    sentiment = RawTickerSentiment(**entry)
                    sentiments.append(sentiment)
                except ValueError as e:
                    self._logger.warning(f"Failed to parse entry {entry}: {e}")
                    # Graceful fallback for this single entry
                    ticker = entry.get("ticker", "UNKNOWN")
                    sentiments.append(
                        RawTickerSentiment(
                            ticker=ticker,
                            sentiment_label="neutral",
                            sentiment_score=0.0,
                            confidence=0.0,
                            rationale=f"(Parse failed: {str(e)})",
                        )
                    )
        
        except json.JSONDecodeError as e:
            self._logger.error(f"JSON parse failed: {e}. Response was: {response_text[:200]}")
            # Fallback: neutral sentiment for all expected tickers
            sentiments = [
                RawTickerSentiment(
                    ticker=ticker,
                    sentiment_label="neutral",
                    sentiment_score=0.0,
                    confidence=0.0,
                    rationale="(JSON parse failed)",
                )
                for ticker in expected_tickers
            ]
        
        return sentiments
    
    
    def _filter_and_rank(self, sentiments: List[RawTickerSentiment]) -> List[FilteredTicker]:
        """
        Filter tickers by confidence threshold, rank by |sentiment_score|.
        
        Args:
            sentiments: Raw sentiments from Gemini
        
        Returns:
            List[FilteredTicker]: Ranked, filtered tickers
        """
        # Filter by confidence
        filtered = [
            s for s in sentiments
            if s.confidence >= self.confidence_threshold
        ]
        
        self._logger.info(
            f"Filtered {len(filtered)} / {len(sentiments)} tickers "
            f"(threshold: {self.confidence_threshold})"
        )
        
        # Rank by absolute sentiment score
        ranked = sorted(
            filtered,
            key=lambda s: abs(s.sentiment_score),
            reverse=True,  # Highest |score| first
        )
        
        # Add rank + absolute_score fields
        result = [
            FilteredTicker(
                ticker=s.ticker,
                sentiment_label=s.sentiment_label,
                sentiment_score=s.sentiment_score,
                confidence=s.confidence,
                rank=idx + 1,
                absolute_score=abs(s.sentiment_score),
            )
            for idx, s in enumerate(ranked)
        ]
        
        return result
    
    
    def _select_top_k(self, filtered: List[FilteredTicker]) -> List[str]:
        """
        Select top K tickers.
        
        Also logs top bullish and top bearish for context.
        
        Args:
            filtered: Ranked, filtered tickers
        
        Returns:
            List[str]: Top K ticker symbols
        """
        selected = filtered[:self.top_k]
        selected_tickers = [t.ticker for t in selected]
        
        # Log top bullish and bearish
        bullish = [t for t in filtered if t.sentiment_label == "bullish"]
        bearish = [t for t in filtered if t.sentiment_label == "bearish"]
        
        top_bullish = [t.ticker for t in bullish[:5]]
        top_bearish = [t.ticker for t in bearish[:5]]
        
        self._logger.info(f"Selected top {len(selected_tickers)} tickers")
        self._logger.info(f"  Top bullish: {top_bullish}")
        self._logger.info(f"  Top bearish: {top_bearish}")
        
        # Log the full result for audit
        result = SeleccionResult(
            selected_tickers=selected_tickers,
            filtered_tickers=filtered,
            total_scanned=len(filtered) + (len(filtered) - len(selected)),  # Approximation
            total_passed_filter=len(filtered),
            filter_threshold=self.confidence_threshold,
            top_bullish=top_bullish,
            top_bearish=top_bearish,
            timestamp=datetime.now().isoformat(),
            gemini_cost_cents=self._estimate_cost(len(filtered) + 100),  # Approximate
        )
        
        # Save to audit log
        self.logger.log_ticker_selection(result)
        
        return selected_tickers
    
    
    # ========================================================================
    # Utility methods
    # ========================================================================
    
    def _create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Split items into batches"""
        return [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
    
    
    def _estimate_cost(self, num_tickers: int, tokens_per_query: int = 500) -> float:
        """
        Estimate cost in cents for Gemini queries.
        
        Args:
            num_tickers: Number of tickers queried
            tokens_per_query: Avg tokens per query
        
        Returns:
            float: Cost in cents
        """
        gemini_cost_per_1k = 0.003  # $0.003 per 1K tokens (2026 pricing)
        total_tokens = (num_tickers / self.batch_size) * tokens_per_query
        total_cost_cents = (total_tokens / 1000) * gemini_cost_per_1k * 100
        return total_cost_cents
    
    
    def _validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid (in stock_data universe)"""
        return ticker in self.stock_data.tickers


# ============================================================================
# TODO / NOTE
# ============================================================================

"""
NOTE: 
    - confidence_threshold=0.6,  # TODO: tune this
    - top_k=120,  # TODO: tune this
    
TODO:
    1. Build out smart model selection in LLMCoordinator
    3. Test batch_size (25 might be too many or too few for Gemini's context)
    4. Tune confidence_threshold (0.6 is reasonable, but validate against data)
    5. Tune top_k (120 is stated target, but could be 50-200)
    6. Build out a stock data fetcher
    7. Thoroughly test
"""