"""Build the fine-tuning set: a strong model writes reports from frozen notes, and only reports
that pass the same grounding checks used for evaluation are kept.

Each kept example is one chat: the writer's system prompt, the writer's input (question and notes),
and the report body as the answer. The body excludes the reference list, which is added from the
real sources after generation, exactly as in the app.

    python -m scholarcrew.finetune.build_dataset --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ..agents.common import ask
from ..agents.writer import SYSTEM, render_references, writer_input
from ..config import Settings
from ..evaluation.metrics import citation_scores, numeric_grounding, structure_score
from ..evaluation.runner import _Recorder
from ..grounding import normalize_citations
from ..llm import make_llm
from ..providers import check_llm
from .freeze import load_frozen

MIN_COVERAGE = 0.7


def quality_problems(body: str, sources: list[dict]) -> list[str]:
    """Why a drafted report body is not good enough to train on. Empty list means keep it."""
    if not body.strip():
        return ["empty"]
    report = f"{body}\n\n{render_references(sources)}\n"
    scores = citation_scores(report, len(sources))
    grounding = numeric_grounding(report, sources)
    problems = []
    if scores["citation_validity"] < 1.0:
        problems.append("cites a source that does not exist")
    if scores["citation_coverage"] < MIN_COVERAGE:
        problems.append(f"only {scores['citation_coverage']:.0%} of statements cited")
    if grounding is not None and grounding < 1.0:
        problems.append("states a figure its cited source does not contain")
    if structure_score(report) < 1.0:
        problems.append("missing title, sections or length")
    return problems


def make_example(row: dict, body: str) -> dict:
    return {
        "id": row["id"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": writer_input(row["question"], row["notes"])},
            {"role": "assistant", "content": body},
        ],
    }


TEACHER_EXTRA = (
    "\nExtra rule: every sentence that states a fact, including the summary and the "
    "limitations, must end with a citation like [1] taken from the notes. Only headings may go "
    "uncited. Do not mention these rules."
)


def draft(llm, row: dict, feedback: list[str] | None = None) -> str:
    """Ask the teacher for a report body. Raises if the model call failed, so a rate limit is not
    mistaken for a bad report.

    The teacher gets a stricter prompt than the writer does. The saved training example still
    uses the plain writer prompt, so the student learns the behavior without being told to.
    """
    system = SYSTEM + TEACHER_EXTRA
    user = writer_input(row["question"], row["notes"])
    if feedback:
        user += (
            "\n\nA previous draft was rejected: " + "; ".join(feedback)
            + ". Write the report again and fix this."
        )
    recorder = _Recorder(llm)
    body = normalize_citations(ask(recorder, system, user)).strip()
    if recorder.errors:
        raise RuntimeError(recorder.errors[0])
    return body


def draft_best(llm, row: dict, attempts: int = 2) -> str:
    """Draft up to `attempts` times, retrying with the reasons the last draft was rejected.
    Stops at the first draft that passes. Returns the last draft otherwise."""
    feedback: list[str] | None = None
    body = ""
    for _ in range(max(1, attempts)):
        body = draft(llm, row, feedback)
        feedback = quality_problems(body, row["sources"])
        if not feedback:
            break
    return body


def build(
    rows: list[dict],
    out_dir: Path,
    teacher: Callable[[dict], str],
    delay: float = 0.0,
    max_consecutive_failures: int = 3,
    on_progress: Callable[[int, int, dict, str], None] | None = None,
) -> dict:
    """Draft and filter every question not already handled. Appends to `train.jsonl` and logs
    discarded drafts in `rejected.jsonl`, so an interrupted run resumes without repeating work."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_path, rejected_path = out_dir / "train.jsonl", out_dir / "rejected.jsonl"
    done = set()
    for path in (kept_path, rejected_path):
        if path.exists():
            done |= {json.loads(line)["id"] for line in path.read_text("utf-8").splitlines() if line}
    counts = {"kept": 0, "rejected": 0, "skipped": 0, "failed": 0, "aborted": False}
    streak = 0
    for i, row in enumerate(rows, 1):
        if row["id"] in done:
            counts["skipped"] += 1
            status = "skipped"
        else:
            try:
                body = teacher(row)
                streak = 0
                problems = quality_problems(body, row["sources"])
                if problems:
                    counts["rejected"] += 1
                    status = "rejected: " + "; ".join(problems)
                    with rejected_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"id": row["id"], "problems": problems, "body": body},
                                           ensure_ascii=False) + "\n")
                else:
                    counts["kept"] += 1
                    status = "kept"
                    with kept_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(make_example(row, body), ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001 - one bad question must not stop the run
                counts["failed"] += 1
                streak += 1
                status = f"failed: {type(exc).__name__}: {str(exc)[:160]}"
            if delay and i < len(rows):
                time.sleep(delay)
        if on_progress:
            on_progress(i, len(rows), row, status)
        if streak >= max_consecutive_failures:
            counts["aborted"] = True
            break
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scholarcrew-build-dataset", description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="teacher model (default: LLM_MODEL)")
    parser.add_argument("--frozen-dir", default="data/frozen/train")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--attempts", type=int, default=2,
                        help="drafts per question; a rejected draft is retried with the reasons")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    rows = load_frozen(Path(args.frozen_dir))
    if not rows:
        print(f"No frozen questions in {args.frozen_dir}. Run the freeze step first.", file=sys.stderr)
        return 2
    base = Settings.from_env()
    settings = replace(base, model=args.model or base.model)
    error = check_llm(settings)
    if error:
        print(f"Cannot reach the model ({settings.model}): {error}", file=sys.stderr)
        return 2
    llm = make_llm(settings)

    def progress(i: int, total: int, row: dict, status: str) -> None:
        print(f"[{i}/{total}] {row['id']}: {status}", file=sys.stderr, flush=True)

    counts = build(rows, Path(args.out_dir), lambda row: draft_best(llm, row, args.attempts),
                   delay=args.delay, on_progress=progress)
    print(json.dumps(counts))
    if counts["aborted"]:
        print("Stopped early after repeated failures (probably a rate limit). "
              "Run the same command again later: finished questions are skipped.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
