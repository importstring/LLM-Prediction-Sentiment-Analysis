from algopkg.data.stock_data import StockData
from algopkg.llm.coordinator import LLMCoordinator
from algopkg.prompts.prompt_manager import PromptManager
from algopkg.research.selection import TickerSelector
from algopkg.research.analysis import ResearchAnalyzer
from algopkg.trading.execution import TradeExecutor
from algopkg.portfolio.portfolio import PortfolioManager
from algopkg.portfolio.learning import LearningEngine
from algopkg.research.logger import ResearchLogger
from algopkg.ui.progress import loading_bar

class Agent:
    def __init__(self, initial_balance: float = 100.0, risk_tolerance: float = 0.02):
        self.balance = initial_balance
        self.risk_tolerance = risk_tolerance

        # Core components
        self.stock_data = StockData()
        self.stock_data.read_stock_data()

        self.portfolio_manager = PortfolioManager()
        self.llms = LLMCoordinator()          # builds { "gpt-4": client, ... }
        self.prompt_manager = PromptManager(self)
        self.research_logger = ResearchLogger()

        # Phase modules
        self.ticker_selector = TickerSelector(self.stock_data, self.llms, self.prompt_manager, self.research_logger)
        self.research_analyzer = ResearchAnalyzer(self.stock_data, self.llms, self.prompt_manager, self.portfolio_manager, self.research_logger)
        self.trade_executor = TradeExecutor(self.portfolio_manager)
        self.learning_engine = LearningEngine(self.portfolio_manager)

        self.tickers = [t for t in self.stock_data.tickers if self.stock_data.is_data_recent(t)]
        self.active_tickers: list[str] = []
        self.plan: str = ""

    def begin(self) -> None:
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

