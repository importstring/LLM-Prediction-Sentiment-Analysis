from __future__ import annotations # TODO: Revise

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List

from algopkg.config import settings
from algopkg.progress_ui.rich_progress import loading_bar  


@dataclass
class Transaction:
    """Representation of a single portfolio transaction."""
    timestamp: str
    action: str          # "buy" | "sell" | etc.
    ticker: str
    shares: float
    price: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioState:
    """Serializable snapshot of portfolio state."""
    timestamp: str
    holdings: Dict[str, Dict[str, Any]]
    pending_stocks: Dict[str, Dict[str, Any]]
    transaction_history: List[Dict[str, Any]]


class Portfolio:
    """
    In-memory portfolio representation with basic persistence.

    Responsibilities:
        - Track current holdings and pending stocks.
        - Track a simple transaction history.
        - Save state under data/databases/portfolio.
    """

    def __init__(self) -> None:
        self.holdings: Dict[str, Dict[str, Any]] = {}
        self.pending_stocks: Dict[str, Dict[str, Any]] = {}
        self.transaction_history: List[Transaction] = []

        self.portfolio_path = settings.portfolio
        self.portfolio_path.mkdir(parents=True, exist_ok=True)

    # ---------- core operations ----------

    def add_to_pending(self, ticker: str, shares: float, reason: str) -> None:
        """Add or update a pending position proposal."""
        self.pending_stocks[ticker] = {
            "shares": float(shares),
            "reason": reason,
        }

    def confirm_portfolio(self) -> None:
        """Move all pending positions into holdings and clear pending."""
        for ticker, info in self.pending_stocks.items():
            existing = self.holdings.get(ticker, {}).get("shares", 0.0)
            total_shares = float(existing) + float(info.get("shares", 0.0))
            self.holdings[ticker] = {
                "shares": total_shares,
                "reason": info.get("reason", "confirmed"),
            }
        self.pending_stocks.clear()

    def record_transaction(
        self,
        action: str,
        ticker: str,
        shares: float,
        price: float,
        meta: Dict[str, Any] | None = None,
    ) -> None:
        """Record a transaction in the in-memory history."""
        tx = Transaction(
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            ticker=ticker,
            shares=float(shares),
            price=float(price),
            meta=meta or {},
        )
        self.transaction_history.append(tx)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Return a lightweight summary of current state."""
        return {
            "holdings": self.holdings,
            "pending": self.pending_stocks,
            "transactions": len(self.transaction_history),
        }

    # ---------- persistence ----------

    def save_state(self) -> bool:
        """
        Save portfolio state and trading information to JSON and a text summary.

        Returns:
            True if save was successful, False otherwise.
        """
        loading_bar.dynamic_update("Saving portfolio state", operation="Portfolio.save_state")
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            state = PortfolioState(
                timestamp=timestamp,
                holdings=self.holdings,
                pending_stocks=self.pending_stocks,
                transaction_history=[asdict(tx) for tx in self.transaction_history],
            )

            state_file = self.portfolio_path / f"portfolio_state_{timestamp}.json"
            with state_file.open("w", encoding="utf-8") as f:
                json.dump(asdict(state), f, indent=2)

            summary = (
                "Portfolio State Save Summary\n"
                "===========================\n"
                f"Timestamp: {timestamp}\n\n"
                "Portfolio State:\n"
                f"- Holdings: {len(self.holdings)}\n"
                f"- Pending Trades: {len(self.pending_stocks)}\n"
                f"- Transaction History: {len(self.transaction_history)}\n\n"
                "Files Saved:\n"
                f"- Portfolio state: {state_file}\n"
            )

            summary_file = self.portfolio_path / f"save_summary_{timestamp}.txt"
            with summary_file.open("w", encoding="utf-8") as f:
                f.write(summary)

            loading_bar.dynamic_update("Portfolio state saved successfully", operation="Portfolio.save_state")
            return True
        except Exception as e:
            logging.error(f"Error saving portfolio state: {e}")
            loading_bar.dynamic_update(f"Error saving portfolio state: {e}", operation="Portfolio.save_state")
            return False

