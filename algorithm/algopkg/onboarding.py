from __future__ import annotations

from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# algorithm/.env (one level up from algopkg/)
ENV_PATH: Final[Path] = Path(__file__).resolve().parents[1] / ".env"


# ---------------------------------------------------------------------------
# Messages (UI text only)
# ---------------------------------------------------------------------------

class OnboardingMessages:
    HEADER = "=== AI Finance Agent Onboarding ==="
    INTRO = (
        "This will create or update a local .env file with your API keys.\n"
        "Keys are stored only on your machine and ignored by git.\n"
    )
    PROVIDERS_INTRO = "Cloud LLM providers (all optional):\n"

    PROMPT_OPENAI = "OpenAI API key (sk-..., leave blank to skip): "
    PROMPT_ANTHROPIC = "Anthropic API key (Claude, leave blank to skip): "
    PROMPT_GEMINI = "Google Gemini API key (leave blank to skip): "
    PROMPT_PERPLEXITY = "Perplexity API key (leave blank to skip): "

    NO_KEYS = (
        "\nNo cloud API keys provided. You can still use local Ollama if installed.\n"
        "Install Ollama from https://ollama.com/download and pull a model, e.g.:\n"
        "  ollama pull llama3"
    )

    @staticmethod
    def success(env_path: Path) -> str:
        return (
            f"\nSaved keys to {env_path}\n"
            "You can now run the agent from the 'algorithm' directory, for example:\n"
            "  cd algorithm\n"
            "  python -m algopkg.agents.agent"
        )


# ---------------------------------------------------------------------------
# Onboarding logic
# ---------------------------------------------------------------------------

class Onboarding:
    """
    CLI helper to configure API keys for cloud LLM providers.

    Writes keys to algorithm/.env, which is ignored by git and loaded in config.py.
    """

    def __init__(self, env_path: Path = ENV_PATH) -> None:
        self.env_path = env_path

    def run(self) -> None:
        self._print_header()
        keys = self._collect_keys()

        if not keys:
            self._print_no_keys_message()
            return

        self._set_or_update_keys(keys)
        self._print_success_message()

    # ----- steps -------------------------------------------------------------

    def _print_header(self) -> None:
        print(OnboardingMessages.HEADER)
        print(OnboardingMessages.INTRO)
        print(OnboardingMessages.PROVIDERS_INTRO)

    def _collect_keys(self) -> dict[str, str]:
        openai_key = input(OnboardingMessages.PROMPT_OPENAI).strip()
        anthropic_key = input(OnboardingMessages.PROMPT_ANTHROPIC).strip()
        gemini_key = input(OnboardingMessages.PROMPT_GEMINI).strip()
        perplexity_key = input(OnboardingMessages.PROMPT_PERPLEXITY).strip()

        keys: dict[str, str] = {}
        if openai_key:
            keys["OPENAI_API_KEY"] = openai_key
        if anthropic_key:
            keys["ANTHROPIC_API_KEY"] = anthropic_key
        if gemini_key:
            keys["GEMINI_API_KEY"] = gemini_key
        if perplexity_key:
            keys["PERPLEXITY_API_KEY"] = perplexity_key

        return keys

    def _print_no_keys_message(self) -> None:
        print(OnboardingMessages.NO_KEYS)

    def _set_or_update_keys(self, kv: dict[str, str]) -> None:
        lines: list[str] = []
        if self.env_path.exists():
            lines = self.env_path.read_text().splitlines()

        filtered = [
            line
            for line in lines
            if not any(line.startswith(f"{key}=") for key in kv.keys())
        ]

        for key, value in kv.items():
            if value:
                filtered.append(f"{key}={value}")

        self.env_path.write_text("\n".join(filtered) + "\n")

    def _print_success_message(self) -> None:
        print(OnboardingMessages.success(self.env_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_onboarding() -> None:
    Onboarding().run()


if __name__ == "__main__":
    run_onboarding()
