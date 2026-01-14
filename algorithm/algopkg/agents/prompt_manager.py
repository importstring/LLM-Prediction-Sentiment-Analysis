from __future__ import annotations

from typing import List, Optional, Dict, Any


class PromptManager:
    """
    Centralized prompt generation for the financial agent.
    
    Responsibilities:
        - Provide clean, consistent prompts for each phase.
        - Handle state formatting (portfolio, tickers, history) for LLM consumption.
        - Maintain single action syntax across all prompts.
    """

    # Role definitions
    ROLE_STOCK_SELECTION = (
        "You are a stock screener identifying 30–120 high-potential stocks from the S&P 500 "
        "for long-term growth (20-year horizon)."
    )
    ROLE_RESEARCH = (
        "You are a fundamental and technical analyst evaluating specific stocks "
        "using financial metrics, industry trends, and valuation frameworks."
    )
    ROLE_REASONING = (
        "You are a portfolio strategist using data analysis, risk assessment, and mathematical reasoning "
        "to synthesize investment decisions."
    )

    # Action reference (consistent syntax)
    ACTION_REFERENCE = """
Available actions (use <action:param> syntax):
  <insight:TICKER>          - Get market insights for a single ticker
  <research:TICKER>         - Get fundamental/technical research for a ticker
  <reason:QUERY>            - Internal reasoning (calculations, synthesis)
  <stockdata:TICKER>        - Fetch OHLCV market data
  <select:TICK1,TICK2,...)>  - Submit final selection (stock selection phase only)

Guidelines:
  - Use <reason:...> for internal math (no LLM cost).
  - Use <insight:...> and <research:...> judiciously (LLM API calls).
  - All tickers must be valid S&P 500 symbols.
  - Final selection: 30–120 tickers.
"""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def get_stock_selection_prompt(self) -> str:
        """
        Generate the prompt for the stock selection phase.
        
        Goal: Filter the S&P 500 down to 30–120 high-potential stocks.
        """
        return f"""
{self.ROLE_STOCK_SELECTION}

Current state:
  - Available tickers: {len(self.agent.tickers)} stocks
    (sample: {", ".join(self.agent.tickers[:5])}...)
  - Current portfolio: {self._format_state("portfolio")}
  - Previous actions: {self._format_state("previous_actions", limit=3)}

Your task:
1. Categorize stocks by sector and risk level.
2. Identify emerging trends and disruptive technologies.
3. Analyze growth potential, competitive advantage, and financial health.
4. Consider macroeconomic factors and industry outlooks.
5. Assign risk-reward scores and eliminate lower-scoring stocks.
6. Maintain diversification across sectors.
7. Narrow down to 30–120 high-potential stocks.
8. Submit your final selection using <select:TICKER1,TICKER2,...>

{self.ACTION_REFERENCE}

Use <reason:...> extensively for internal analysis. Use <insight:...> and <research:...> for critical validation.
Only submit your final <select:...> when you have your curated list of 30–120 tickers.
"""

    def get_research_prompt(
        self,
        tickers: List[str] | None = None,
        max_tickers: int = 15,
        focus_areas: Optional[List[str]] = None,
    ) -> str:
        """
        Generate the prompt for the research phase.
        
        Goal: Deep dive into a curated set of tickers with comprehensive analysis.
        
        Args:
            tickers: Tickers to analyze. Defaults to agent.active_tickers.
            max_tickers: Limit to avoid prompt bloat.
            focus_areas: Optional list of analysis dimensions (e.g., ["valuation", "growth"]).
        """
        tickers = (tickers or self.agent.active_tickers)[:max_tickers]
        focus = (
            "\n  ".join(focus_areas)
            if focus_areas
            else "Fundamental analysis, Technical analysis, Valuation, Risk assessment"
        )

        return f"""
{self.ROLE_RESEARCH}

Tickers to analyze: {", ".join(tickers)}

Focus areas:
  {focus}

Framework:
1. Fundamental Analysis:
   - Financial statements, ratios (P/E, P/B, ROE, debt-to-equity, margins)
   - Competitive position, economic moat, market share

2. Technical Analysis:
   - Price trends, volume patterns, support/resistance, momentum indicators
   - Potential breakout points and risk zones

3. Industry & Macro:
   - Sector performance, cyclicality, growth drivers
   - Macroeconomic headwinds and tailwinds

4. Qualitative Factors:
   - Management quality, governance, R&D spending, innovation pipeline

5. Risk Assessment:
   - Market, financial, operational, regulatory risks
   - ESG impact on long-term value

6. Valuation:
   - Intrinsic value (DCF, comparables, sum-of-parts)
   - Margin of safety vs. current price

7. Decision:
   - Buy, sell, or hold? Specify position sizing rationale.

Current portfolio: {self._format_state("portfolio")}
Previous plan: {self._format_state("plan", limit=5)}

{self.ACTION_REFERENCE}

Prioritize <reason:...> for analysis. Use <stockdata:...> to fetch recent prices/volume.
Use <insight:...> and <research:...> to validate critical assumptions.
"""

    def get_reasoning_prompt(self, context: str, max_history: int = 5) -> str:
        """
        Generate the prompt for the reasoning phase.
        
        Goal: Synthesize data into clear, actionable insights.
        
        Args:
            context: Current context or analysis results.
            max_history: Number of previous actions to include.
        """
        return f"""
{self.ROLE_REASONING}

Current context:
{context}

Portfolio state: {self._format_state("portfolio")}
Recent actions:
{self._format_state("previous_actions", limit=max_history)}

Analysis checklist:
  ✓ Market conditions and trends
  ✓ Technical indicators and price momentum
  ✓ Fundamental metrics and valuation
  ✓ Risk factors and catalysts
  ✓ Portfolio impact and diversification
  ✓ Entry/exit timing and position sizing
  ✓ Long-term strategic alignment

{self.ACTION_REFERENCE}

Provide clear, step-by-step reasoning. Use <reason:...> for math and synthesis.
Fetch <stockdata:...> if you need recent prices or volume.
Conclude with actionable next steps.
"""

    # ---------- internals ----------

    def _format_state(self, key: str, limit: Optional[int] = None) -> str:
        """
        Format agent state into readable text for the prompt.
        
        Prevents dumping raw Python structures into the LLM.
        """
        if key == "portfolio":
            portfolio = getattr(self.agent, "portfolio", {})
            if not portfolio:
                return "Empty"
            items = list(portfolio.items())[:limit] if limit else list(portfolio.items())
            return ", ".join([f"{k}:{v}" for k, v in items])

        if key == "previous_actions":
            actions = getattr(self.agent, "previous_actions", [])
            if not actions:
                return "None"
            items = actions[-limit:] if limit else actions
            return "\n  ".join([str(a) for a in items])

        if key == "plan":
            plan = getattr(self.agent, "plan", "")
            if not plan:
                return "No plan yet."
            lines = str(plan).split("\n")[:limit] if limit else str(plan).split("\n")
            return "\n  ".join(lines)

        return "N/A"
