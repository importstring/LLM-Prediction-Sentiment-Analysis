from algopkg.agents.agent import Agent
from algopkg.progress_ui.rich_progress import make_cli_reporter
from algopkg.progress_ui.status import PhaseStatus, PhaseName

def run_cli():
    reporter, sink = make_cli_reporter()
    agent = Agent(status_reporter=reporter)

    with sink:
        agent.main()

if __name__ == "__main__":
    run_cli()

