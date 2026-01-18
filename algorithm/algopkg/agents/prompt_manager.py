from __future__ import annotations

from typing import Any, List, Optional, Dict


class PromptManager:
    """
    Centralized prompt generation for the financial agent.

    This is a stripped-down manager that only knows how to render
    concrete prompt templates for the different phases (sentiment,
    research, ensemble, portfolio construction, learning, etc.).

    Responsibilities:
      - Provide clean, consistent prompts for each phase.
      - Handle state formatting (portfolio, tickers, history) for LLM consumption.
      - Maintain consistent JSON-only output instructions.
    """

    # ======================================================================
    # Phase 1 – Sentiment prompts
    # ======================================================================

    def ticker_sentiment_batch(
        self,
        tickers: List[str],
    ) -> str:
        batch_size = len(tickers)
        tickers_block = ", ".join(tickers)
        return f"""
You are a financial sentiment analyst analyzing stock market sentiment.

I will give you a list of tickers. For EACH ticker, output your sentiment assessment.

TICKERS TO ANALYZE ({batch_size} total):
{tickers_block}

For each ticker, analyze:
1. Recent market momentum (price trend, volume)
2. Sector rotation trends
3. Company fundamentals (if you have knowledge)
4. Macroeconomic factors affecting this sector
5. Technical chart patterns (if you have data)

OUTPUT INSTRUCTIONS:
- Output ONLY a valid JSON array. No markdown, no explanation, no extra text.
- Each element must have exactly these keys:
  - "ticker": the stock symbol
  - "sentiment_label": MUST be exactly one of: "bullish", "neutral", "bearish"
  - "sentiment_score": a float from -1.0 (bearish) to +1.0 (bullish). Examples:
    * -0.9 to -1.0: strongly bearish
    * -0.5 to -0.9: moderately bearish
    * -0.2 to -0.5: slightly bearish
    * -0.2 to +0.2: neutral
    * +0.2 to +0.5: slightly bullish
    * +0.5 to +0.9: moderately bullish
    * +0.9 to +1.0: strongly bullish
  - "confidence": a float from 0.0 to 1.0 indicating how confident you are in this sentiment
    * 0.0-0.3: low confidence (uncertain, conflicting signals)
    * 0.3-0.7: moderate confidence (some strong signals, some weak)
    * 0.7-1.0: high confidence (strong consensus of multiple factors)
  - "rationale": a 1-2 sentence explanation of your sentiment

EXAMPLE OUTPUT (valid JSON array):
[
  {{"ticker": "AAPL", "sentiment_label": "bullish", "sentiment_score": 0.65, "confidence": 0.78, "rationale": "Strong tech sector momentum, robust earnings, high institutional buying pressure."}},
  {{"ticker": "XYZ", "sentiment_label": "bearish", "sentiment_score": -0.42, "confidence": 0.55, "rationale": "Sector headwinds, declining volume, mixed guidance from competitors."}},
  {{"ticker": "ABC", "sentiment_label": "neutral", "sentiment_score": 0.05, "confidence": 0.61, "rationale": "Balanced bull and bear signals; awaiting earnings catalysts."}}
]

Now analyze the tickers above. Output ONLY the JSON array, nothing else.
""".strip()

    def ticker_sentiment_single(
        self,
        ticker: str,
        sector: str,
        recent_events: str,
    ) -> str:
        return f"""
You are a financial sentiment analyst.

Analyze this stock and provide a sentiment assessment:

TICKER: {ticker}
SECTOR: {sector}
RECENT EVENTS / CONTEXT: {recent_events}

Assess sentiment by considering:
1. Price momentum (short-term trend)
2. Sector health and rotation
3. Company-specific catalysts (earnings, product launches, regulatory news)
4. Macroeconomic backdrop
5. Valuation context (relative to peers)

Output ONLY a JSON object (not an array) with these exact keys:
- "ticker": the symbol
- "sentiment_label": one of "bullish", "neutral", "bearish"
- "sentiment_score": -1.0 to +1.0
- "confidence": 0.0 to 1.0
- "rationale": brief explanation

Example:
{{"ticker": "AAPL", "sentiment_label": "bullish", "sentiment_score": 0.72, "confidence": 0.85, "rationale": "Positive sentiment from AI chip demand, strong cash flow, and institutional accumulation."}}

Analyze {ticker} now. Output ONLY the JSON object, nothing else.
""".strip()

    # ======================================================================
    # Phase 2 – Ensemble & financial/agentic synthesis
    # ======================================================================

    def ensemble_sentiment_interpretation(
        self,
        ticker: str,
        llm_sentiments_table: str,
        disagreement_level: str,
    ) -> str:
        return f"""
You are synthesizing sentiment from multiple AI analysts.

TICKER: {ticker}

INDIVIDUAL LLM SENTIMENTS:
{llm_sentiments_table}

Example table:
| Model      | Sentiment  | Score | Confidence |
|------------|-----------|-------|------------|
| ChatGPT    | bullish   | 0.68  | 0.82       |
| Claude     | bullish   | 0.75  | 0.88       |
| Perplexity | neutral   | 0.15  | 0.71       |
| Gemini     | bullish   | 0.62  | 0.79       |

DISAGREEMENT LEVEL: {disagreement_level}
(Low = high consensus, High = models strongly disagree)

Interpret this panel:
1. What is the consensus sentiment? (bullish/neutral/bearish)
2. How confident should we be in this consensus?
3. What are the key points of agreement / disagreement?
4. If models disagree, which factors caused the split?
5. Any red flags or noteworthy outliers?

Output a JSON object:
{{
  "ensemble_label": "bullish/neutral/bearish",
  "ensemble_score": (float, -1 to 1),
  "consensus_confidence": (float, 0 to 1),
  "key_agreement": "what models agree on",
  "key_disagreement": "where models split",
  "interpretation": "2-3 sentences on what this ensemble sentiment means"
}}

Analyze the ensemble above. Output ONLY the JSON object.
""".strip()

    def financial_metrics_synthesis(
        self,
        ticker: str,
        metrics_data: str,
    ) -> str:
        return f"""
You are a quantitative analyst synthesizing financial metrics.

TICKER: {ticker}

FINANCIAL METRICS DATA:
{metrics_data}

Example:
- Current Price: $150.25
- Implied Volatility (IV): 28%
- Black-Scholes Fair Value: $148.50
- IV Surface Skew: Normal
- 3-month Price Prediction (ML model): $158.20 ± $12.00
- Risk Metrics:
  * Sharpe Ratio: 0.95
  * VaR (95%, 1-day): -2.3%
  * Efficient Frontier Position: Mid-risk, mid-return
- Valuation:
  * P/E Ratio: 22x (vs sector 20x)
  * PEG Ratio: 1.1
  * Price-to-Book: 3.2

Evaluate:
1. Is the stock fairly valued (B-S vs market price)?
2. What is the risk-adjusted upside potential?
3. Do the ML predictions align with fundamental value?
4. How does volatility compare to historical and peer levels?
5. Does this fit your risk tolerance and time horizon?

Output a JSON object:
{{
  "valuation_assessment": "fairly valued / undervalued / overvalued",
  "valuation_score": (0-100, higher = better value),
  "risk_assessment": "low / moderate / high",
  "upside_potential": "score 0-100 based on ML predictions and fundamentals",
  "downside_risk": "score 0-100, with VaR and stress test results",
  "combined_financial_score": (0-100, this becomes the 'X' in final ranking),
  "key_insights": "2-3 sentence summary of financial health"
}}

Synthesize the metrics above. Output ONLY the JSON object.
""".strip()

    def agentic_reflection_and_validation(
        self,
        ticker: str,
        agent_outputs: str,
        portfolio_constraints: str,
    ) -> str:
        return f"""
You are an orchestrator reviewing specialized agent analyses.

TICKER: {ticker}

AGENT OUTPUTS (from 5 specialized agents):
{agent_outputs}

Example:
- Pricing Agent: "B-S fair value $148, current $150 (slight premium). IV normal. Recommend neutral on entry price."
- Risk Agent: "VaR acceptable, Greeks balanced. Mild gamma exposure. Good for core portfolio."
- Portfolio Agent: "Efficient frontier: efficient position. Correlations low with existing holdings. Add to optimize Sharpe."
- ML Agent: "Model predicts $158 ± $12. High confidence. Aligns with fundamental upside."
- Data Quality Agent: "All data fresh as of today. IV surface liquid. No data quality concerns."

PORTFOLIO CONSTRAINTS:
{portfolio_constraints}

Example:
- Maximum position size: 5% of portfolio
- Risk tolerance: moderate (target volatility 12%)
- Sector allocation: Tech capped at 25%
- Rebalance frequency: monthly

REFLECTION QUESTIONS:
1. Do all agents agree this ticker is a good candidate?
2. Are there any red flags or concerns across the five perspectives?
3. Does adding this ticker respect the portfolio constraints?
4. What is the agents' collective confidence level (0-100)?
5. Should this ticker be included, excluded, or marked "conditional"?

Output a JSON object:
{{
  "agent_consensus": "strong agree / mild agree / neutral / mild disagree / strong disagree",
  "consensus_score": (0-100, higher = more unified agreement),
  "key_concerns": "list any red flags raised by agents",
  "portfolio_fit": "excellent / good / acceptable / poor",
  "constraint_violations": "any hard constraints violated? List them or 'none'",
  "recommendation": "include / exclude / conditional (with condition)",
  "agentic_analysis_score": (0-100, this becomes the 'Y' in final ranking),
  "rationale": "2-3 sentences on agent consensus"
}}

Reflect on the agent outputs above. Output ONLY the JSON object.
""".strip()

    # ======================================================================
    # Phase 3 – Combined ranking & portfolio construction
    # ======================================================================

    def combined_ranking_and_portfolio_construction(
        self,
        ticker_scores_table: str,
        weighting_strategy: str,
    ) -> str:
        return f"""
You are building a final AI portfolio from scored candidates.

CANDIDATE TICKER SCORES:
{ticker_scores_table}

Example:
| Ticker | X (Finance) | Y (Agentic) | Z (Sentiment) | Current Weight |
|--------|------------|------------|---------------|----------------|
| AAPL   | 85         | 78         | 82            | 0              |
| MSFT   | 92         | 81         | 75            | 0              |
| GOOGL  | 88         | 85         | 80            | 0              |
| NVDA   | 79         | 82         | 88            | 0              |
| TSLA   | 72         | 68         | 65            | 0              |

WEIGHTING STRATEGY:
{weighting_strategy}

Example:
- Method: Weighted average with dynamic adjustment
- Weights:
  * X (Financial): 40% (fundamentals matter most)
  * Y (Agentic): 30% (agent reasoning + constraints)
  * Z (Sentiment): 30% (ensemble LLM consensus)
- Risk Target: 12% portfolio volatility
- Position Size Limits: max 5% per stock, max 25% per sector
- Rebalance: hold for 1 month then reassess

PORTFOLIO CONSTRUCTION:
1. Calculate W = 0.4*X + 0.3*Y + 0.3*Z for each ticker
2. Sort by W (descending)
3. Allocate positions to hit risk target while respecting constraints
4. Ensure no sector exceeds 25%, no stock exceeds 5%
5. Output final portfolio weights and rationale

Output a JSON object:
{{
  "combined_scores": {{
    "AAPL": {{"W": 82.1, "rank": 1}},
    "MSFT": {{"W": 84.2, "rank": 1}}
  }},
  "portfolio_allocation": {{
    "AAPL": 0.04,
    "MSFT": 0.05
  }},
  "portfolio_metrics": {{
    "expected_return": "X%",
    "target_volatility": "12%",
    "sharpe_ratio_estimate": 1.2,
    "sector_allocation": {{"tech": 0.25, "finance": 0.20}}
  }},
  "rationale": "Brief summary of portfolio construction logic"
}}

Build the portfolio. Output ONLY the JSON object.
""".strip()

    # ======================================================================
    # Phase 4 – Learning / meta-analysis
    # ======================================================================

    def learning_meta_analysis(
        self,
        historical_trades: str,
        actual_outcomes: str,
        current_weights: str,
    ) -> str:
        return f"""
You are analyzing the performance of the AI portfolio agent to improve it.

RECENT TRADE HISTORY:
{historical_trades}

Example:
- Run 1 (Jan 10): Selected AAPL (W=85), MSFT (W=82). Result: +3.2% return.
- Run 2 (Jan 11): Selected NVDA (W=88), TSLA (W=65). Result: -1.1% return.
- Run 3 (Jan 12): Selected GOOGL (W=80), META (W=78). Result: +0.8% return.

ACTUAL MARKET OUTCOMES:
{actual_outcomes}

Example:
- AAPL returned +4.2% (vs predicted +3%)
- MSFT returned +2.1% (vs predicted +2.5%)
- NVDA returned -2.5% (vs predicted +1.5%)
- TSLA returned -3.1% (vs predicted -0.5%)
- GOOGL returned +1.2% (vs predicted +0.8%)
- META returned +2.8% (vs predicted +1%)

CURRENT WEIGHTS:
{current_weights}

(X: Finance 40%, Y: Agentic 30%, Z: Sentiment 30%)

ANALYSIS:
1. Which component (X, Y, or Z) predicted best?
2. Were high-scored picks actually good? (Accuracy)
3. Were low-scored picks actually bad? (Negative predictive value)
4. Did ensemble sentiment outperform individual LLMs?
5. Should we reweight the components?
6. Any systematic biases or failure modes?

Output a JSON object:
{{
  "component_accuracy": {{
    "X_financial": {{"accuracy": 0.62, "trend": "underperforming"}},
    "Y_agentic": {{"accuracy": 0.65, "trend": "stable"}},
    "Z_sentiment": {{"accuracy": 0.71, "trend": "strong"}}
  }},
  "recommended_weight_adjustment": {{
    "X": 0.35,
    "Y": 0.25,
    "Z": 0.40
  }},
  "key_insights": "List 2-3 main insights from this analysis",
  "systematic_biases": "Any patterns? e.g., 'tends to overweight growth'",
  "next_iteration_changes": "What to change for next run"
}}

Analyze performance and recommend improvements. Output ONLY the JSON object.
""".strip()
