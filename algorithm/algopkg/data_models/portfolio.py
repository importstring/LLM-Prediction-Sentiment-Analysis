from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from algopkg.config import settings
from algopkg.progress_ui.rich_progress import loading_bar


logger = logging.getLogger(__name__)


# ---------- data models ----------


@dataclass(slots=True)
class Transaction:
    """Representation of a single portfolio transaction."""
    timestamp: str
    action: str          # "buy" | "sell" | etc.
    ticker: str
    shares: float
    price: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioState:
    """Serializable snapshot of portfolio state."""
    timestamp: str
    holdings: Dict[str, Dict[str, Any]]
    pending_stocks: Dict[str, Dict[str, Any]]
    transaction_history: List[Dict[str, Any]]


# ---------- domain object ----------


class Portfolio:
    """
    In-memory portfolio representation with basic persistence.

    Responsibilities:
        - Track current holdings and pending stocks.
        - Track a simple transaction history.
        - Save state under data/databases/portfolio.
    """

    def __init__(self, portfolio_path: Optional[Path] = None) -> None:
        self.holdings: Dict[str, Dict[str, Any]] = {}
        self.pending_stocks: Dict[str, Dict[str, Any]] = {}
        self.transaction_history: List[Transaction] = []

        self.portfolio_path = portfolio_path or settings.portfolio
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
            existing_shares = float(self.holdings.get(ticker, {}).get("shares", 0.0))
            new_shares = float(info.get("shares", 0.0))
            total_shares = existing_shares + new_shares

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
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record a transaction in the in-memory history."""
        tx = Transaction(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            ticker=ticker,
            shares=float(shares),
            price=float(price),
            meta=dict(meta) if meta is not None else {},
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

    def _build_state(self, timestamp: str) -> PortfolioState:
        """Build a serializable PortfolioState snapshot."""
        return PortfolioState(
            timestamp=timestamp,
            holdings=self.holdings,
            pending_stocks=self.pending_stocks,
            transaction_history=[asdict(tx) for tx in self.transaction_history],
        )

    def _write_json(self, state: PortfolioState, state_file: Path) -> None:
        with state_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)

    def _write_summary(self, timestamp: str, state_file: Path, summary_file: Path) -> None:
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
        with summary_file.open("w", encoding="utf-8") as f:
            f.write(summary)

    def save_state(self) -> bool:
        """
        Save portfolio state and trading information to JSON and a text summary.

        Returns:
            True if save was successful, False otherwise.
        """
        loading_bar.dynamic_update(
            "Saving portfolio state",
            operation="Portfolio.save_state",
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        state_file = self.portfolio_path / f"portfolio_state_{timestamp}.json"
        summary_file = self.portfolio_path / f"save_summary_{timestamp}.txt"

        try:
            state = self._build_state(timestamp)
            self._write_json(state, state_file)
            self._write_summary(timestamp, state_file, summary_file)
        except Exception:
            logger.exception("Error saving portfolio state")
            loading_bar.dynamic_update(
                "Error saving portfolio state (see logs for details)",
                operation="Portfolio.save_state",
            )
            return False

        loading_bar.dynamic_update(
            "Portfolio state saved successfully",
            operation="Portfolio.save_state",
        )
        return True
