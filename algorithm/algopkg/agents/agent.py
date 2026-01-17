from __future__ import annotations

import logging
from typing import List, Dict, Any

from algopkg.data_models.stock_data import StockData
from algopkg.data_models.portfolio import PortfolioManager

from algopkg.llm_clients.llm_base import BaseLLMClient as LLMCoordinator  

from algopkg.agents.prompt_manager import PromptManager
from algopkg.agents.ticker_selector import TickerSelector
from algopkg.agents.research_analyzer import ResearchAnalyzer
from algopkg.agents.trade_executor import TradeExecutor
from algopkg.agents.learning_engine import LearningEngine
from algopkg.agents.research_logger import ResearchLogger

from algopkg.progress_ui.rich_progress import loading_bar


class Agent:
    """
    High-level orchestrator for the AI-finance research pipeline.

    Stages (run in main()):
      Stage 1: Cheap global scan (Gemini only)
        - Get a normalized sentiment score for all S&P 500 stocks
        - Penalize low-confidence outputs if possible
        - Keep top K by score (e.g. top 50–120 + bottom 50–120)
      Stage 2: Expensive deep-dive research (all LLMs + finance)
        - Query ChatGPT, Claude, Perplexity, etc.
        - Run financial analysis
        - Run agentic analysis with reasoning steps
      Stage 3: Scoring & Ranking
        - Financial Analysis: composite of financial metrics
        - Agentic Analysis: qualitative + constraint-aware verdict (0–100)
        - LLM Sentiment: sentiment from all LLMs (0–100)
        - Combined Score: weighted average of the three scores
      Stage 4: Portfolio construction
        - Build a portfolio allocation plan based on ranked list + risk constraints
        - Size positions
        - Save final weights / recommendations
    """

    def __init__(self, initial_balance: float = 100.0, risk_tolerance: float = 0.02):
        self.balance = initial_balance
        self.risk_tolerance = risk_tolerance

        # Core components
        self.stock_data = StockData()  # loads or refreshes data in __init__
        self.portfolio_manager = PortfolioManager()

        # LLM registry (one client per provider)
        self.llms = LLMCoordinator(
            enable_sentiment=True,  # or False, depending on default
        )

        self.research_logger = ResearchLogger()
        self.prompt_manager = PromptManager(self)

        # Phase modules
        self.ticker_selector = TickerSelector(
            stock_data=self.stock_data,
            llms=self.llms,
            prompt_manager=self.prompt_manager,
            logger=self.research_logger,
        )
        self.research_analyzer = ResearchAnalyzer(
            stock_data=self.stock_data,
            llms=self.llms,
            prompt_manager=self.prompt_manager,
            portfolio_manager=self.portfolio_manager,
            logger=self.research_logger,
        )
        self.trade_executor = TradeExecutor(self.portfolio_manager)
        self.learning_engine = LearningEngine(self.portfolio_manager)

        # State
        self.tickers: List[str] = [
            t for t in self.stock_data.tickers
            if self.stock_data.is_data_recent(t)
        ]
        self.active_tickers: List[str] = []
        self.plan: str = ""
        self.latest_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def main(self) -> None:
        """
        Run the full multi-phase research pipeline.
        """
        with loading_bar:
            phases = [
                ("Stock Selection", self.pick_tickers),
                ("Research Analysis", self.research_and_insight),
                ("Trade Execution", self.execute_trades),
                ("Learning Phase", self.learn),
            ]
            for name, fn in phases:
                op = f"begin.{name.lower().replace(' ', '_')}"
                loading_bar.dynamic_update(f"Starting {name}", operation=op)
                fn()
                loading_bar.dynamic_update(f"Completed {name}", operation=op)

        self._generate_comprehensive_report()
        self.finalize_execution()
        loading_bar.dynamic_update("✅ Agent execution completed", operation="begin")

    # Backwards-compatible entry point
    def begin(self) -> None:
        self.main()

    # ------------------------------------------------------------------
    # Phase 1: Stock Selection
    # ------------------------------------------------------------------

    def pick_tickers(self) -> List[str]:
        """
        Phase 1: Select active_tickers for this run using TickerSelector.
        """
        loading_bar.dynamic_update(
            "Starting stock selection process",
            operation="pick_tickers",
        )
        self.active_tickers = self.ticker_selector.select(self.tickers)
        return self.active_tickers

    # ------------------------------------------------------------------
    # Phase 2: Research & Insight
    # ------------------------------------------------------------------

    def research_and_insight(self) -> Dict[str, Any]:
        """
        Phase 2: Run deep research + sentiment analysis and build a
        hypothetical AI portfolio via ResearchAnalyzer.
        """
        loading_bar.dynamic_update(
            "Starting research and insight analysis",
            operation="research_and_insight",
        )
        if not self.active_tickers:
            logging.warning("No active_tickers set before research_and_insight")
            self.latest_results = {}
            return self.latest_results

        self.latest_results = self.research_analyzer.analyze(self.active_tickers)
        return self.latest_results

    # ------------------------------------------------------------------
    # Phase 3: Trade Execution (logging only)
    # ------------------------------------------------------------------

    def execute_trades(self) -> None:
        """
        Phase 3: Persist trade recommendations for offline review/backtests.
        Does not execute real trades.
        """
        loading_bar.dynamic_update(
            "Saving trade recommendations",
            operation="execute_trades",
        )
        self.trade_executor.save_recommendations()

    # ------------------------------------------------------------------
    # Phase 4: Learning
    # ------------------------------------------------------------------

    def learn(self) -> None:
        """
        Phase 4: Learn from past recommendations/trades and adjust
        high-level strategy parameters (e.g. risk_tolerance).
        """
        loading_bar.dynamic_update(
            "Learning from recent trades",
            operation="learn",
        )
        self.learning_engine.learn_from_history(self)

    # ------------------------------------------------------------------
    # Reporting / finalization (placeholders)
    # ------------------------------------------------------------------

    def _generate_comprehensive_report(self) -> None:
        """
        Summarize the run (rankings, rationales, portfolio suggestion, etc.).
        Implement this to emit markdown/JSON/logs as needed.
        """
        # TODO: implement reporting logic
        pass

    def finalize_execution(self) -> None:
        """
        Final cleanup / persistence hook after the pipeline completes.
        """
        # TODO: implement any cleanup/persistence hooks
        pass
