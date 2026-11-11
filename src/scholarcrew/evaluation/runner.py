"""Run the crew over a question set, score every report, and aggregate the results."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime
from importlib import resources
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import Settings
from ..graph import stream_crew
from .judge import judge_faithfulness, judge_relevance
from .metrics import deterministic_scores, extract_claims, numeric_grounding

METRICS = [
    ("faithfulness", "Faithfulness", "cited claims the sources support (judged)"),
    ("numeric_grounding", "Numeric grounding", "statistics that appear in the source they cite"),
    ("relevance", "Relevance", "how well the report answers the question (judged)"),
    ("citation_validity", "Citation validity", "citation markers that point at a real source"),
    ("citation_coverage", "Citation coverage", "substantive statements that carry a citation"),
    ("source_utilization", "Source utilization", "retrieved sources the report cites"),
    ("structure", "Structure", "title, sections, references, length"),
]

Generator = Callable[[str], dict]


def load_questions(n: int | None = None, topic: str | None = None) -> list[dict]:
    text = resources.files("scholarcrew.evaluation").joinpath("questions.json").read_text("utf-8")
    items = json.loads(text)
    if topic:
        items = [q for q in items if q["topic"].lower() == topic.lower()]
    return items[:n] if n else items


class LLMFailure(RuntimeError):
    """A model call failed while generating a report, so the report is degraded."""


class _Recorder:
    """Passes calls through to the model and remembers any that fail.

    The agents catch model errors and fall back to a degraded report so the app keeps working.
    That is right for users but wrong for measurement: a report written after a rate-limit error
    must not be scored as if the system had worked.
    """

    def __init__(self, llm):
        self._llm, self.errors = llm, []

    def invoke(self, *args, **kwargs):
        try:
            return self._llm.invoke(*args, **kwargs)
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            raise


def make_generator(llm: BaseChatModel, settings: Settings, **tools: Callable) -> Generator:
    """Wrap the real crew as question -> {report, sources, log, seconds}.

    Raises LLMFailure if any model call failed, so the run records the question as failed.
    """

    def generate(question: str) -> dict:
        started = time.perf_counter()
        recorder = _Recorder(llm)
        report, sources, log = "", [], []
        for node, update in stream_crew(question, recorder, settings, **tools):
            log += [f"{node}: {line}" for line in update.get("log", [])]
            sources = update.get("sources", sources)
            report = update.get("report", report)
        if recorder.errors:
            raise LLMFailure(f"{len(recorder.errors)} model call(s) failed, first: {recorder.errors[0]}")
        return {
            "report": report,
            "sources": sources,
            "log": log,
            "seconds": time.perf_counter() - started,
        }

    return generate


def score_row(row: dict, judge: BaseChatModel) -> dict:
    """Attach scores to a generated row. Judge failures become None, never silent zeros."""
    report, sources = row["report"], row["sources"]
    scores: dict = dict(deterministic_scores(report, len(sources)))
    scores["numeric_grounding"] = numeric_grounding(report, sources) if report else None
    claims = extract_claims(report)
    if report and not claims:
        # Nothing in the report cites a source, so nothing in it is verifiably grounded. Scoring
        # this as "no data" would let uncited reports escape the average and flatter the system.
        scores["faithfulness"] = 0.0
    else:
        scores["faithfulness"] = judge_faithfulness(judge, claims, sources) if report else None
    scores["relevance"] = judge_relevance(judge, row["question"], report) if report else None
    return {**row, "scores": scores, "n_claims_checked": len(claims)}


def aggregate(rows: list[dict]) -> dict:
    """Mean per metric over rows that were scored; None values are excluded and counted."""
    ok = [r for r in rows if not r.get("error")]
    summary: dict = {"questions": len(rows), "failed": len(rows) - len(ok)}
    for key, *_ in METRICS:
        values = [r["scores"][key] for r in ok if r["scores"].get(key) is not None]
        summary[key] = sum(values) / len(values) if values else None
        summary[f"{key}_n"] = len(values)
    times = [r["seconds"] for r in ok]
    summary["mean_seconds"] = sum(times) / len(times) if times else None
    summary["mean_sources"] = sum(len(r["sources"]) for r in ok) / len(ok) if ok else None
    return summary


def run(
    items: list[dict],
    generate: Generator,
    judge: BaseChatModel,
    delay: float = 0.0,
    on_progress: Callable[[int, int, dict], None] | None = None,
    meta: dict | None = None,
    max_consecutive_failures: int = 3,
) -> dict:
    """Generate and score every question. Stops early if the provider keeps failing.

    Stopping early matters on free tiers: once a daily quota is exhausted every further question
    fails, and continuing only wastes time and any remaining quota.
    """
    meta = dict(meta or {})
    rows, streak = [], 0
    for i, item in enumerate(items, 1):
        row = {**item, "report": "", "sources": [], "log": [], "seconds": 0.0, "error": None}
        try:
            row.update(generate(item["question"]))
            row = score_row(row, judge)
            streak = 0
        except Exception as exc:  # noqa: BLE001 - one bad question must not abort the whole run
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            row["scores"] = {}
            streak += 1
        rows.append(row)
        if on_progress:
            on_progress(i, len(items), row)
        if streak >= max_consecutive_failures:
            meta["aborted"] = (
                f"stopped after {streak} failures in a row at question {i} of {len(items)}; "
                f"last error: {row['error']}"
            )
            break
        if delay and i < len(items):
            time.sleep(delay)
    return {"meta": meta, "summary": aggregate(rows), "rows": rows}


def rescore(result: dict, judge: BaseChatModel, meta: dict | None = None) -> dict:
    """Score saved outputs again (for example with a different judge) without regenerating."""
    rows = [
        score_row(r, judge) if not r.get("error") else r
        for r in result["rows"]
    ]
    return {"meta": {**result.get("meta", {}), **(meta or {})}, "summary": aggregate(rows), "rows": rows}


def resume_merge(previous: dict, fresh: dict) -> dict:
    """Combine a finished-or-partial run with a re-run of its failed or missing questions.

    Keeps every successful row from `previous`, adds the rows from `fresh`, restores the original
    question order, and recomputes the summary. Free tiers cap daily tokens, so completing a run
    over more than one session is a normal case, not an edge case.
    """
    kept = [r for r in previous["rows"] if not r.get("error")]
    order = {q["id"]: i for i, q in enumerate(load_questions())}
    rows = sorted(kept + fresh["rows"], key=lambda r: order.get(r["id"], len(order)))
    meta = {**previous.get("meta", {}), **fresh.get("meta", {}), "resumed": True}
    if not any(r.get("error") for r in rows):
        meta.pop("aborted", None)
    return {"meta": meta, "summary": aggregate(rows), "rows": rows}


def check_thresholds(summary: dict, minimums: dict[str, float]) -> list[str]:
    """Return a message for each metric below its minimum (or missing). Empty list means pass."""
    problems = []
    for key, minimum in minimums.items():
        value = summary.get(key)
        if value is None:
            problems.append(f"{key}: no score available")
        elif value < minimum:
            problems.append(f"{key}: {value:.2f} is below the minimum {minimum:.2f}")
    return problems


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def to_markdown(result: dict) -> str:
    meta, summary, rows = result["meta"], result["summary"], result["rows"]
    lines = ["## Evaluation results", ""]
    if meta:
        lines.append(
            f"Generator: `{meta.get('model', '?')}` | Judge: `{meta.get('judge_model', '?')}` | "
            f"Questions: {summary['questions']} ({summary['failed']} failed) | {meta.get('date', '')}"
        )
        lines.append("")
    if meta.get("aborted"):
        lines += [f"> **Run aborted:** {meta['aborted']}", ""]
    lines += ["| Metric | Score | What it measures |", "|---|---|---|"]
    for key, label, desc in METRICS:
        n = summary.get(f"{key}_n", 0)
        lines.append(f"| {label} | {_pct(summary.get(key))} (n={n}) | {desc} |")
    if summary.get("mean_seconds") is not None:
        timing = (
            f"Mean time per report: {summary['mean_seconds']:.0f}s. "
            f"Mean sources retrieved: {summary['mean_sources']:.1f}."
        )
        lines += ["", timing]
    lines += ["", "<details><summary>Per-question scores</summary>", "",
              "| ID | Topic | Faithful | Relevant | Cite valid | Cite coverage | Time |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r.get("scores", {})
        if r.get("error"):
            lines.append(f"| {r['id']} | {r['topic']} | failed: {r['error'][:60]} | | | | |")
        else:
            lines.append(
                f"| {r['id']} | {r['topic']} | {_pct(s.get('faithfulness'))} | {_pct(s.get('relevance'))} | "
                f"{_pct(s.get('citation_validity'))} | {_pct(s.get('citation_coverage'))} | {r['seconds']:.0f}s |"
            )
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def save(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path, md_path = out_dir / f"eval-{stamp}.json", out_dir / f"eval-{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(to_markdown(result), encoding="utf-8")
    return json_path, md_path
