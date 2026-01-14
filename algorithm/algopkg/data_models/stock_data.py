"""
Stock data retrieval and caching using yfinance and local CSV files.

IMPORTANT:
    This implementation is intended for personal, research, and educational use only.
    Yahoo Finance data accessed via the `yfinance` library is generally not licensed
    for direct commercial use. For any commercial or production deployment, this
    module MUST be replaced or modified to use a properly licensed market data
    provider in accordance with its terms of service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf 


STOCK_DATA_PATH = Path("Data/Stock-Data")


@dataclass(frozen=True)
class StockDataConfig:
    data_path: Path = STOCK_DATA_PATH
    recency_tolerance_days: int = 2
    start: str = "2015-01-01"
    end: str = "2100-01-01"


class StockData:
    """
    Local stock data loader and refresher backed by Yahoo Finance via yfinance.

    Responsibilities:
        - Ensure the local stock data directory is populated.
        - Refresh data if it is stale beyond a configured tolerance.
        - Provide accessors for single and multiple tickers.
    """

    def __init__(self, config: StockDataConfig | None = None) -> None:
        self.config = config or StockDataConfig()
        self.data_path = self.config.data_path

        self._ensure_data_directory()
        self._maybe_refresh_data()

        self.stock_data: Dict[str, pd.DataFrame] = self._read_stock_data()
        if not self.stock_data:
            raise RuntimeError(
                f"No stock data found under {self.data_path}. "
                "Ensure yfinance download completed successfully."
            )

        self.tickers: List[str] = list(self.stock_data.keys())

    # ---------- public API ----------

    def get_stock_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self.stock_data.get(ticker)
            if df is not None and not df.empty:
                data[ticker] = df
        return data

    def get_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        df = self.stock_data.get(ticker)
        if df is None or df.empty:
            return None
        return df

    def is_data_recent(self, ticker: str, tolerance_days: Optional[int] = None) -> bool:
        """
        Check whether data for a ticker is up to date within the given tolerance.
        """
        tolerance = (
            tolerance_days
            if tolerance_days is not None
            else self.config.recency_tolerance_days
        )
        df = self.get_ticker_data(ticker)
        if df is None or df.empty:
            return False

        last_date = df.index.max().normalize()
        current_date = pd.Timestamp.today().normalize()
        return (current_date - last_date).days <= tolerance

    # ---------- internals ----------

    def _ensure_data_directory(self) -> None:
        """
        Ensure the data directory exists and is not completely empty.

        If the directory does not exist, it is created.
        If it exists but contains no CSV files, a download will be required.
        """
        self.data_path.mkdir(parents=True, exist_ok=True)
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            logging.warning(
                "Stock data directory is empty. "
                "Initial download from Yahoo Finance is required."
            )
            self._download_all_sp500()

    def _maybe_refresh_data(self) -> None:
        """
        Refresh data from yfinance if it appears stale.

        Uses a small sample of tickers (or all if few) to check recency.
        """
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            return

        sample_files = csv_files[:10]
        stale = False
        tolerance = self.config.recency_tolerance_days

        for fp in sample_files:
            try:
                df = pd.read_csv(fp, parse_dates=["Date"])
                last_date = df["Date"].max().normalize()
                current_date = pd.Timestamp.today().normalize()
                if (current_date - last_date).days > tolerance:
                    stale = True
                    break
            except Exception as e:
                logging.warning(f"Failed to inspect recency for {fp}: {e}")
                stale = True
                break

        if stale:
            logging.info("Local stock data appears stale. Refreshing from Yahoo Finance.")
            self._download_all_sp500()

    def _read_stock_data(self) -> Dict[str, pd.DataFrame]:
        stock_data: Dict[str, pd.DataFrame] = {}
        try:
            csv_files = list(self.data_path.glob("*.csv"))
            for file_path in csv_files:
                ticker = file_path.stem
                df = pd.read_csv(file_path, parse_dates=["Date"])
                df.set_index("Date", inplace=True)
                stock_data[ticker] = df
            return stock_data
        except Exception as e:
            logging.error(f"Error reading stock data: {e}")
            return {}

    def _download_all_sp500(self) -> None:
        """
        Download S&P 500 constituents' OHLCV data with yfinance and save one CSV per ticker.

        Note:
            Uses Wikipedia to obtain the current list of S&P 500 companies.
        """
        tickers = self._get_sp500_tickers()
        if not tickers:
            raise RuntimeError("Failed to retrieve S&P 500 tickers for download.")

        data = yf.download(
            tickers,
            start=self.config.start,
            end=self.config.end,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )  # [web:184][web:188][web:193]

        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers:
                if ticker not in data:
                    continue
                df = data[ticker].dropna(how="all")
                if df.empty:
                    continue
                df.to_csv(self.data_path / f"{ticker}.csv", index_label="Date")
        else:
            # Single-ticker or unexpected shape; fallback: do per-ticker loop
            for ticker in tickers:
                hist = yf.download(
                    ticker,
                    start=self.config.start,
                    end=self.config.end,
                    auto_adjust=False,
                )
                if hist.empty:
                    continue
                hist.to_csv(self.data_path / f"{ticker}.csv", index_label="Date")

    @staticmethod
    def _get_sp500_tickers() -> List[str]:
        """
        Retrieve the current list of S&P 500 tickers from Wikipedia.
        """
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        return table["Symbol"].tolist()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sd = StockData()
    print(f"Loaded {len(sd.tickers)} tickers from {sd.data_path}")
    sample = sd.tickers[:5]
    for t in sample:
        df = sd.get_ticker_data(t)
        last_date = df.index.max().date() if df is not None else "N/A"
        print(t, "rows:", len(df) if df is not None else 0, "last:", last_date)
