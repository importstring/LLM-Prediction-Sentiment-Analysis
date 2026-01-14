from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project-root/
DATA_ROOT = PROJECT_ROOT / "data"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

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

settings = Settings()

