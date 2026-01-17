from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from algopkg.utils.paths import find_project_root


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = find_project_root(Path(__file__))

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


settings: Final[Settings] = Settings()

# ---------------------------------------------------------------------------
# API keys / secrets
# ---------------------------------------------------------------------------

load_dotenv(ENV_PATH)

OPENAI_API_KEY: Final[str | None] = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: Final[str | None] = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY: Final[str | None] = os.getenv("GEMINI_API_KEY")
PERPLEXITY_API_KEY: Final[str | None] = os.getenv("PERPLEXITY_API_KEY")
