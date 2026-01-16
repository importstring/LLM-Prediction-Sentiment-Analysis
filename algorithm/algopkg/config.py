from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import os
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]  # project-root/
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
SCRIPTS_ROOT: Final[Path] = PROJECT_ROOT / "scripts"
ENV_PATH: Final[Path] = PROJECT_ROOT / "algorithm" / ".env"


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_root: Path = DATA_ROOT

    conversations: Path = DATA_ROOT / "conversations"
    stock_data: Path = DATA_ROOT / "stock_data"
    databases: Path = DATA_ROOT / "databases"
    portfolio: Path = DATA_ROOT / "databases" / "portfolio"
    info_tickers: Path = DATA_ROOT / "info" / "tickers"
    rankings: Path = DATA_ROOT / "rankings"

    api_keys_root: Path = SCRIPTS_ROOT / "llm_insights" / "api_keys"


settings: Final[Settings] = Settings()


# ---------------------------------------------------------------------------
# API keys / secrets
# ---------------------------------------------------------------------------

# Load .env if present (for local development)
load_dotenv(ENV_PATH)

OPENAI_API_KEY: Final[str | None] = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: Final[str | None] = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY: Final[str | None] = os.getenv("GEMINI_API_KEY")
PERPLEXITY_API_KEY: Final[str | None] = os.getenv("PERPLEXITY_API_KEY")
