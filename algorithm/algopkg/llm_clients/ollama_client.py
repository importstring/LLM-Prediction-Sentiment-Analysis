from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import requests


@dataclass(frozen=True)
class OllamaConfig:
    """Configuration for Ollama client. Last updated Jan 2026."""

    base_url: str = "http://127.0.0.1:11434"
    default_model: str = "phi3:mini"  # Phi-3 Mini 3.8B parameters
    default_temperature: float = 0.5
    default_num_predict: int = 512

    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def generate_url(self) -> str:
        return f"{self.base_url}/api/generate"

    def list_models_url(self) -> str:
        return f"{self.base_url}/api/tags"


class OllamaClient:
    """
    High-level Ollama client.

    Responsibilities:
    - Call /api/chat and /api/generate on a local Ollama instance.
    - Provide simple chat() and generate() helpers.
    - Support a configurable default model (phi3:mini by default).
    """

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()

    # ---------- public API ----------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        stream: bool = False,
    ) -> Tuple[str, bool]:
        """
        Call the Ollama /api/chat endpoint.

        messages: list of {"role": "user" | "assistant" | "system", "content": "..."}.
        Returns (assistant_text, success_flag).
        """
        temperature = float(temperature if temperature is not None else self.config.default_temperature)
        num_predict = int(num_predict if num_predict is not None else self.config.default_num_predict)

        payload = {
            "model": self.config.default_model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
            "stream": stream,
        }

        try:
            resp = requests.post(self.config.chat_url(), json=payload, timeout=60)
            resp.raise_for_status()

            if stream:
                # For now, collect the streamed responses into a single string.
                content = []
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = line.decode("utf-8")
                        # downstream code can parse JSON if needed; kept simple here
                        content.append(data)
                    except Exception:
                        continue
                return "\n".join(content), True

            data = resp.json()
            return data["message"]["content"], True
        except Exception as e:
            logging.error(f"Ollama chat() failed: {e}")
            return f"Error: {e}", False

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        stream: bool = False,
    ) -> Tuple[str, bool]:
        """
        Call the Ollama /api/generate endpoint.

        Returns (response_text, success_flag).
        """
        temperature = float(temperature if temperature is not None else self.config.default_temperature)
        num_predict = int(num_predict if num_predict is not None else self.config.default_num_predict)

        payload = {
            "model": self.config.default_model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
            "stream": stream,
        }

        try:
            resp = requests.post(self.config.generate_url(), json=payload, timeout=60)
            resp.raise_for_status()

            if stream:
                chunks: List[str] = []
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = line.decode("utf-8")
                        chunks.append(data)
                    except Exception:
                        continue
                return "\n".join(chunks), True

            data = resp.json()
            return data["response"], True
        except Exception as e:
            logging.error(f"Ollama generate() failed: {e}")
            return f"Error: {e}", False

    def list_models(self) -> Tuple[List[Dict[str, Union[str, int]]], bool]:
        """
        List models available in the local Ollama instance.

        Returns (models_json, success_flag).
        """
        try:
            resp = requests.get(self.config.list_models_url(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", data.get("data", [])), True
        except Exception as e:
            logging.error(f"Ollama list_models() failed: {e}")
            return [], False


if __name__ == "__main__":
    client = OllamaClient()
    text, ok = client.generate("Summarize the CAPM model in finance in 3 bullet points.")
    if ok:
        print(text)
    else:
        print("Error:", text)

