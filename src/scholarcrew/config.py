"""Runtime settings, read from environment variables (or a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader so we don't need another dependency."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    max_steps: int = 8
    reports_dir: Path = Path("reports")

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", "ollama"),
            model=os.getenv("LLM_MODEL", "llama3.2:3b"),
            max_steps=int(os.getenv("MAX_STEPS", "8")),
            reports_dir=Path(os.getenv("REPORTS_DIR", "reports")),
        )
