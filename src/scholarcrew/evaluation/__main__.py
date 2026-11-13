"""Command line: python -m scholarcrew.evaluation [--quick] [--model M] [--judge-model J]."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

from ..config import Settings
from ..llm import make_llm_pool
from ..providers import check_llm
from .runner import (
    check_thresholds,
    load_questions,
    make_generator,
    rescore,
    resume_merge,
    run,
    save,
    to_markdown,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scholarcrew-eval", description=__doc__)
    p.add_argument("--n", type=int, help="only the first N questions")
    p.add_argument("--quick", action="store_true", help="shortcut for --n 5")
    p.add_argument("--topic", help="only questions of one topic, e.g. RAG")
    p.add_argument("--model", help="generator model (default: LLM_MODEL from .env)")
    p.add_argument("--judge-model", help="judge model (default: same as the generator)")
    p.add_argument("--delay", type=float, default=3.0, help="seconds to pause between questions")
    p.add_argument("--out-dir", default="eval_results", help="where to save results")
    p.add_argument("--rescore", metavar="FILE", help="re-judge saved outputs without regenerating")
    p.add_argument("--resume", metavar="FILE",
                   help="finish a partial run: re-run only questions that failed or are missing")
    for key in ("faithfulness", "relevance", "citation_validity"):
        p.add_argument(f"--min-{key.replace('_', '-')}", type=float, metavar="X",
                       help=f"exit with an error if mean {key} is below X (0 to 1)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    base = Settings.from_env()
    gen_settings = replace(base, model=args.model or base.model)
    judge_settings = replace(base, model=args.judge_model or gen_settings.model)
    meta = {
        "model": gen_settings.model,
        "judge_model": judge_settings.model,
        "date": date.today().isoformat(),
    }

    judge = make_llm_pool(judge_settings, temperature=0.0)
    generator_llm = make_llm_pool(gen_settings)
    for label, settings, model in (("generator", gen_settings, generator_llm), ("judge", judge_settings, judge)):
        error = check_llm(settings, model)
        if error:
            print(f"Cannot reach the {label} model ({settings.model}): {error}", file=sys.stderr)
            return 2

    if args.rescore:
        previous = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        result = rescore(previous, judge, meta={"judge_model": judge_settings.model})
    else:
        items = load_questions(5 if args.quick else args.n, args.topic)
        if not items:
            print("No questions matched.", file=sys.stderr)
            return 2
        previous = None
        if args.resume:
            previous = json.loads(Path(args.resume).read_text(encoding="utf-8"))
            for key in ("model", "judge_model"):
                if previous["meta"].get(key) != meta[key]:
                    print(f"Cannot resume: {key} was {previous['meta'].get(key)!r} but is now "
                          f"{meta[key]!r}. Mixing models would make the result meaningless.",
                          file=sys.stderr)
                    return 2
            done = {r["id"] for r in previous["rows"] if not r.get("error")}
            items = [q for q in items if q["id"] not in done]
            if not items:
                print("Nothing to resume: every selected question already succeeded.", file=sys.stderr)
                print(to_markdown(previous))
                return 0

        def progress(i: int, total: int, row: dict) -> None:
            s = row.get("scores", {})
            detail = row["error"] or (
                f"faith={_fmt(s.get('faithfulness'))} rel={_fmt(s.get('relevance'))} "
                f"cite={_fmt(s.get('citation_validity'))} {row['seconds']:.0f}s"
            )
            print(f"[{i}/{total}] {row['id']}: {detail}", file=sys.stderr, flush=True)

        generate = make_generator(generator_llm, gen_settings)
        result = run(items, generate, judge, delay=args.delay, on_progress=progress, meta=meta)
        if previous:
            result = resume_merge(previous, result)

    json_path, md_path = save(result, Path(args.out_dir))
    print(to_markdown(result))
    print(f"Saved {json_path} and {md_path}", file=sys.stderr)

    summary = result["summary"]
    if summary["failed"] == summary["questions"]:
        return 2
    minimums = {
        key: getattr(args, f"min_{key}")
        for key in ("faithfulness", "relevance", "citation_validity")
        if getattr(args, f"min_{key}") is not None
    }
    problems = check_thresholds(summary, minimums)
    for message in problems:
        print(f"THRESHOLD FAILED: {message}", file=sys.stderr)
    return 1 if problems else 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


if __name__ == "__main__":
    raise SystemExit(main())
