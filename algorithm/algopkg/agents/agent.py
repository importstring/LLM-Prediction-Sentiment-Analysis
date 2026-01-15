from algopkg.data_models.stock_data import StockData
from algopkg.data_models.portfolio import PortfolioManager

from algopkg.llm_clients.coordinator import LLMCoordinator

from algopkg.agents.prompt_manager import PromptManager
from algopkg.agents.ticker_selector import TickerSelector
from algopkg.agents.research_analyzer import ResearchAnalyzer
from algopkg.agents.trade_executor import TradeExecutor
from algopkg.agents.learning_engine import LearningEngine
from algopkg.agents.research_logger import ResearchLogger

from algopkg.progress_ui.rich_progress import loading_bar


import logging
from typing import List, Dict, Any


class Agent:
    """
    High-level orchestrator for the AI-finance research pipeline.

    Phases (run in main()):
      0. Strategic Planning        -> plan_actions()
      1. Stock Selection           -> pick_tickers()
      2. Research & Insight        -> research_and_insight()
      3. Trade Execution (logging) -> execute_trades()
      4. Learning Phase            -> learn()
    """

    def __init__(self, initial_balance: float = 100.0, risk_tolerance: float = 0.02):
        self.balance = initial_balance
        self.risk_tolerance = risk_tolerance

        # Core components
        self.stock_data = StockData()
        self.stock_data.read_stock_data()

        self.portfolio_manager = PortfolioManager()
        self.llms = LLMCoordinator()  # builds {"gpt-4": ..., "claude": ..., ...}
        self.prompt_manager = PromptManager(self)
        self.research_logger = ResearchLogger()

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
        Run the full 5-phase research pipeline.
        Previously called `begin()`.
        """
        with loading_bar:
            phases = [
                ("Strategic Planning", self.plan_actions),
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

    # Using main.py but just in case for old logic
    def begin(self) -> None:
        self.main()

    # ------------------------------------------------------------------
    # Phase 0: Strategic Planning
    # ------------------------------------------------------------------

    def plan_actions(self) -> None:
        """
        Phase 0: Generate a high-level plan of actions for this run
        using PromptManager + planner LLM.
        """
        loading_bar.dynamic_update("Planning actions", operation="plan_actions")
        loading_bar.dynamic_update("Generating plan", operation="Planning")

        prompt = self.prompt_manager.get_planning_prompt()
        # TODO: switch to planner via LLMCoordinator if desired
        self.plan = self.llms.plan(prompt)  # or self.ollama.make_plan(prompt)

        loading_bar.dynamic_update("Plan generated", operation="Planning")
        loading_bar.dynamic_update("Actions planned", operation="plan_actions")

    # ------------------------------------------------------------------
    # Phase 1: Stock Selection
    # ------------------------------------------------------------------

    def pick_tickers(self) -> List[str]:
        """
        Phase 1: Select active_tickers for this run using TickerSelector.
        """
        loading_bar.dynamic_update("Starting stock selection process", operation="pick_tickers")
        self.active_tickers = self.ticker_selector.select(self.tickers)
        return self.active_tickers

    # ------------------------------------------------------------------
    # Phase 3: Research & Insight
    # ------------------------------------------------------------------

    def research_and_insight(self) -> Dict[str, Any]:
        """
        Phase 3: Run deep research + sentiment analysis and build a
        hypothetical AI portfolio via ResearchAnalyzer.
        """
        loading_bar.dynamic_update("Starting research and insight analysis",
                                   operation="research_and_insight")
        if not self.active_tickers:
            logging.warning("No active_tickers set before research_and_insight")
            self.latest_results = {}
            return self.latest_results

        self.latest_results = self.research_analyzer.analyze(self.active_tickers)
        return self.latest_results

    # ------------------------------------------------------------------
    # Phase 4: Trade Execution (logging only)
    # ------------------------------------------------------------------

    def execute_trades(self) -> None:
        """
        Phase 4: Persist trade recommendations for offline review/backtests.
        Does not execute real trades.
        """
        loading_bar.dynamic_update("Saving trade recommendations", operation="execute_trades")
        self.trade_executor.save_recommendations()

    # ------------------------------------------------------------------
    # Phase 5: Learning
    # ------------------------------------------------------------------

    def learn(self) -> None:
        """
        Phase 5: Learn from past recommendations/trades and adjust
        high-level strategy parameters (e.g. risk_tolerance).
        """
        loading_bar.dynamic_update("Learning from recent trades", operation="learn")
        self.learning_engine.learn_from_history(self)
