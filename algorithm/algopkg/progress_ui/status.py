from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class PhaseName(str, Enum):
    PLANNING = "planning"
    SELECTION = "selection"
    RESEARCH = "research"
    EXECUTION = "execution"
    LEARNING = "learning"

@dataclass
class PhaseStatus:
    name: PhaseName
    label: str          # "Stock Selection", ...
    progress: float     # 0.0–1.0
    message: str        # "Analyzing 12/80 tickers", ...

class AgentStatusReporter:
    """Bridge between Agent and UIs (CLI & frontend)."""

    def __init__(self, sink):
        self.sink = sink  # anything with .update(status: PhaseStatus)

    def update_phase(self, status: PhaseStatus) -> None:
        self.sink.update(status)
