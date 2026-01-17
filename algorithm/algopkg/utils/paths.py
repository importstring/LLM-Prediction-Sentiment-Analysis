from __future__ import annotations

from pathlib import Path
from typing import Final


PROJECT_FOLDER_NAME: Final[str] = "LLM-Prediction-Sentiment-Analysis"


def find_project_root(
    start: Path | None = None,
    project_name: str = PROJECT_FOLDER_NAME,
) -> Path:
    """
    Walk up from `start` until a directory named `project_name` is found.

    Args:
        start: Path to start from (defaults to this file's location).
        project_name: Name of the repo root directory.

    Returns:
        Path to the project root.

    Raises:
        RuntimeError: If the project root cannot be found.
    """
    if start is None:
        start = Path(__file__).resolve()

    current = start.resolve()
    for parent in (current, *current.parents):
        if parent.name == project_name:
            return parent

    raise RuntimeError(
        f"Could not find project root named {project_name!r} starting from {start}"
    )
