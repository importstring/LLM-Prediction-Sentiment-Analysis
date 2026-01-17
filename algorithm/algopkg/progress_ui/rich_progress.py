"""CLI progress reporting utilities using Rich.

Provides:
- LoadingBar: context-managed sink that renders phase progress with a
  spinner and progress bar.
- make_cli_reporter: helper to construct an AgentStatusReporter wired to a
  LoadingBar for CLI use.
"""

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console
from status import PhaseStatus, AgentStatusReporter


class LoadingBar:
    """Sink that renders phase status updates using a Rich progress bar."""

    def __init__(self):
        """Initialize the Rich console, progress instance, and task registry."""
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            expand=True,
        )
        self.tasks = {}  # phase_name -> task_id

    def __enter__(self):
        """Start the progress rendering context."""
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Stop the progress rendering context."""
        self.progress.stop()

    def update(self, status: PhaseStatus) -> None:
        """Create or update the progress bar for the given phase status.

        Args:
            status: Current phase status, including name, label, message,
                and progress (0.0–1.0).
        """
        key = status.name.value
        if key not in self.tasks:
            self.tasks[key] = self.progress.add_task(status.label, total=100)
        task_id = self.tasks[key]
        self.progress.update(
            task_id,
            description=f"{status.label} • {status.message}",
            completed=int(status.progress * 100),
        )


def make_cli_reporter() -> AgentStatusReporter:
    """Create an AgentStatusReporter backed by a LoadingBar.

    Returns:
        A tuple of (AgentStatusReporter, LoadingBar) wired for CLI use.
    """
    sink = LoadingBar()
    return AgentStatusReporter(sink), sink
