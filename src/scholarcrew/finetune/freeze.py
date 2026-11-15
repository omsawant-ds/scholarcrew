"""Freeze the writer's inputs: run the researcher and analyst once per question, save the result.

The writer is the stage being fine-tuned, so everything before it is held fixed. Each question
becomes one JSON file holding its sources and the analyst's notes. Later steps (building the
training set, comparing writers) read these files and never repeat the retrieval or analysis,
which saves free-tier tokens and means every writer sees identical input.

    python -m scholarcrew.finetune.freeze --split train
    python -m scholarcrew.finetune.freeze --split eval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from importlib import resources
from pathlib import Path

from ..agents import make_analyst, make_researcher
from ..config import Settings
from ..evaluation.runner import LLMFailure, _Recorder, load_questions
from ..llm import make_llm_pool
from ..providers import check_llm
from ..tools import search_arxiv, search_web


def load_train_questions() -> list[dict]:
    text = resources.files("scholarcrew.finetune").joinpath("train_questions.json").read_text("utf-8")
    return json.loads(text)


def _norm(question: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()


def overlap(train: list[dict], held_out: list[dict]) -> list[str]:
    """Training questions that duplicate a held-out question. The two sets must stay disjoint,
    otherwise the evaluation would measure memorization instead of skill."""
    held = {_norm(q["question"]) for q in held_out}
    return [q["id"] for q in train if _norm(q["question"]) in held]


def freeze_one(
    item: dict,
    llm,
    arxiv_search: Callable = search_arxiv,
    web_search: Callable = search_web,
) -> dict:
    """Run researcher and analyst for one question. Raises if the result would be degraded."""
    recorder = _Recorder(llm)
    state: dict = {"question": item["question"]}
    state.update(make_researcher(recorder, arxiv_search, web_search)(state))
    if not state["sources"]:
        raise LLMFailure("the researcher found no sources")
    state.update(make_analyst(recorder)(state))
    if recorder.errors:
        raise LLMFailure(f"{len(recorder.errors)} model call(s) failed, first: {recorder.errors[0]}")
    if state["notes"].startswith("Analysis unavailable"):
        raise LLMFailure("the analyst returned nothing")
    return {**item, "sources": state["sources"], "notes": state["notes"]}


def freeze(
    items: list[dict],
    out_dir: Path,
    freeze_item: Callable[[dict], dict],
    delay: float = 0.0,
    max_consecutive_failures: int = 3,
    on_progress: Callable[[int, int, dict, str], None] | None = None,
) -> dict:
    """Freeze every question that is not already on disk, so an interrupted run can simply be
    started again. Stops early when the provider keeps failing (a spent daily quota)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"saved": 0, "skipped": 0, "failed": 0, "aborted": False}
    streak = 0
    for i, item in enumerate(items, 1):
        path = out_dir / f"{item['id']}.json"
        if path.exists():
            counts["skipped"] += 1
            status = "skipped"
        else:
            try:
                path.write_text(
                    json.dumps(freeze_item(item), indent=2, ensure_ascii=False), encoding="utf-8"
                )
                counts["saved"] += 1
                streak, status = 0, "saved"
            except Exception as exc:  # noqa: BLE001 - one bad question must not stop the run
                counts["failed"] += 1
                streak += 1
                status = f"failed: {type(exc).__name__}: {str(exc)[:160]}"
            if status == "saved" and delay and i < len(items):
                time.sleep(delay)
        if on_progress:
            on_progress(i, len(items), item, status)
        if streak >= max_consecutive_failures:
            counts["aborted"] = True
            break
    return counts


def load_frozen(directory: Path) -> list[dict]:
    """Every frozen question in a directory, in file-name order."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scholarcrew-freeze", description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=["train", "eval"], required=True,
                        help="train: the fine-tuning questions. eval: the 30 evaluation questions")
    parser.add_argument("--n", type=int, help="only the first N questions of the split")
    parser.add_argument("--model", help="model for the researcher and analyst (default: LLM_MODEL)")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds to pause between questions")
    parser.add_argument("--out-dir", default="data/frozen", help="parent folder; a split subfolder is used")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    train, held_out = load_train_questions(), load_questions()
    clash = overlap(train, held_out)
    if clash:
        print(f"Training questions duplicate evaluation questions: {clash}", file=sys.stderr)
        return 2
    items = (train if args.split == "train" else held_out)[: args.n]

    base = Settings.from_env()
    settings = replace(base, model=args.model or base.model)
    llm = make_llm_pool(settings)
    error = check_llm(settings, llm)
    if error:
        print(f"Cannot reach the model ({settings.model}): {error}", file=sys.stderr)
        return 2

    def progress(i: int, total: int, item: dict, status: str) -> None:
        print(f"[{i}/{total}] {item['id']}: {status}", file=sys.stderr, flush=True)

    counts = freeze(
        items, Path(args.out_dir) / args.split, lambda item: freeze_one(item, llm),
        delay=args.delay, on_progress=progress,
    )
    print(json.dumps(counts))
    if counts["aborted"]:
        print("Stopped early after repeated failures (probably a rate limit). "
              "Run the same command again later: finished questions are skipped.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
