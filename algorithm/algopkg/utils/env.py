from __future__ import annotations

import os
from typing import Final

from algopkg.utils.paths import ENV_PATH


def _load_dotenv_if_needed() -> None:
    if getattr(_load_dotenv_if_needed, "_loaded", False):
        return

    if ENV_PATH.is_file():
        try:
            for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            pass

    setattr(_load_dotenv_if_needed, "_loaded", True)


def get_env(key: str) -> str:
    _load_dotenv_if_needed()
    value = os.getenv(key, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {key} is not set or empty")
    return value
