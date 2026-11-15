"""Command-line entry point: scholarcrew "your research question"."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .config import Settings
from .graph import stream_crew
from .llm import make_llm
from .tools import save_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scholarcrew", description=__doc__)
    parser.add_argument("question", help="the research question to investigate")
    parser.add_argument("--strict", action="store_true",
                        help="keep only statements whose cited source has matching text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Windows consoles default to cp1252 and crash on characters like U+2011 that models emit.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    settings = replace(Settings.from_env(), strict_evidence=args.strict)
    if not settings.all_keys:
        print("No API key found. Put your Groq key in .env as LLM_API_KEY "
              "(free at https://console.groq.com/keys).", file=sys.stderr)
        return 2
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
