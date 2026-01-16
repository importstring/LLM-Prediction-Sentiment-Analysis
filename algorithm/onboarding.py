from __future__ import annotations
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _set_or_update_keys(env_path: Path, kv: dict[str, str]) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    # Remove existing entries for keys we will write
    filtered = [
        ln for ln in lines
        if not any(ln.startswith(f"{k}=") for k in kv.keys())
    ]

    for k, v in kv.items():
        if v:
            filtered.append(f"{k}={v}")

    env_path.write_text("\n".join(filtered) + "\n")


def run_onboarding() -> None:
    print("=== AI Finance Agent Onboarding ===")
    print("This will create/update a local .env file with your API keys.")
    print("Keys are stored only on your machine and ignored by git.\n")

    openai_key = input("Enter your OpenAI API key (or leave blank to skip): ").strip()
    perplexity_key = input("Enter your Perplexity API key (or leave blank to skip): ").strip()

    kv: dict[str, str] = {}
    if openai_key:
        kv["OPENAI_API_KEY"] = openai_key
    if perplexity_key:
        kv["PERPLEXITY_API_KEY"] = perplexity_key

    if not kv:
        print("No keys provided. Nothing to do.")
        return

    _set_or_update_keys(ENV_PATH, kv)
    print(f"\nSaved keys to {ENV_PATH}")
    print("You can now run the agent from the 'algorithm' directory, e.g.:")
    print("  python -m algopkg.agents.agent")


if __name__ == "__main__":
    run_onboarding()