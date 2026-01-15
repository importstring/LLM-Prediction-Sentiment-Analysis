from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console
from .status import PhaseStatus, AgentStatusReporter

class RichProgressSink:
    def __init__(self):
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
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.progress.stop()

    def update(self, status: PhaseStatus) -> None:
        key = status.name.value
        if key not in self.tasks:
            self.tasks[key] = self.progress.add_task(
                status.label, total=100
            )
        task_id = self.tasks[key]
        self.progress.update(
            task_id,
            description=f"{status.label} • {status.message}",
            completed=int(status.progress * 100),
        )

def make_cli_reporter() -> AgentStatusReporter:
    sink = RichProgressSink()
    return AgentStatusReporter(sink), sink

