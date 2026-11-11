"""Command-line entry point: scholarcrew "your research question"."""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .graph import stream_crew
from .llm import make_llm
from .tools import save_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scholarcrew", description=__doc__)
    parser.add_argument("question", help="the research question to investigate")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    llm = make_llm(settings)
    report = ""
    for node, update in stream_crew(args.question, llm, settings):
        for line in update.get("log", []):
            print(f"  [{node}] {line}", file=sys.stderr)
        report = update.get("report", report)

    if not report:
        print("No report was produced. Check your LLM settings in .env.", file=sys.stderr)
        return 1
    path = save_report(args.question, report, settings.reports_dir)
    print(report)
    print(f"\nSaved to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
