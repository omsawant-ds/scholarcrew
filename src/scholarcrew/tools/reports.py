"""Save finished reports to disk as Markdown."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _slug(text: str, limit: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].strip("-") or "report"


def save_report(question: str, markdown: str, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"{stamp}-{_slug(question)}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
