"""Compare the tuned writer with its untuned base and a copy-the-notes baseline.

Every writer reads the same frozen inputs (the sources and notes saved by `freeze`), so any
difference in the reports comes from the writer alone. Two steps:

1. generate (needs a GPU, run in the Kaggle notebook): the base model and the base model with the
   LoRA adapter each write reports for the evaluation questions. The same 4-bit weights are used
   for both; the adapter is simply switched off for the base.

       python -m scholarcrew.finetune.compare generate --adapter outputs/writer-lora

2. score (a laptop is enough): every report is scored with the evaluation pipeline and its
   independent judge, and the writers are compared in one table.

       python -m scholarcrew.finetune.compare score --generations generations.jsonl \\
           --judge-model qwen/qwen3.8-27b

Generated reports are saved one line at a time, and scores likewise, so either step can be
stopped and started again; finished work is never repeated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from datetime import date
from pathlib import Path

from ..agents.writer import SYSTEM, render_references, writer_input
from ..evaluation.runner import METRICS, aggregate, score_row
from ..grounding import normalize_citations
from .baselines import copy_notes_writer
from .freeze import load_frozen

COPY = "copy-notes"
VARIANTS = ("base", "tuned")
# Qwen's own recommended settings for answering without its thinking mode.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def strip_thinking(text: str) -> str:
    """Remove a thinking block, finished or cut off, if the model wrote one."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL).strip()


def make_report(body: str, sources: list[dict]) -> str:
    """A generated body with the real reference list added, exactly as the writer agent does."""
    return f"{normalize_citations(strip_thinking(body)).strip()}\n\n{render_references(sources)}\n"


def key_of(row: dict) -> tuple[str, str, int]:
    return row["id"], row["variant"], row["sample"]


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def append_line(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---- step 1: generation (GPU) ------------------------------------------------------------------


def generate(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    frozen = load_frozen(Path(args.frozen))
    if args.n:
        frozen = frozen[: args.n]
    if not frozen:
        print(f"No frozen questions in {args.frozen}.", file=sys.stderr)
        return 2
    out = Path(args.out)
    done = {key_of(r) for r in read_lines(out)}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        ),
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    total = len(frozen) * args.samples * len(VARIANTS)
    count = 0
    for row in frozen:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": writer_input(row["question"], row["notes"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        for sample in range(args.samples):
            for variant in VARIANTS:
                count += 1
                item = {"id": row["id"], "variant": variant, "sample": sample}
                if key_of(item) in done:
                    continue
                set_seed(args.seed + sample)  # base and tuned share the seed for a given sample
                started = time.perf_counter()
                with (model.disable_adapter() if variant == "base" else nullcontext()), torch.no_grad():
                    tokens = model.generate(
                        **batch, max_new_tokens=args.max_new_tokens, do_sample=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **SAMPLING,
                    )
                body = tokenizer.decode(tokens[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
                seconds = time.perf_counter() - started
                append_line(out, {**item, "body": body, "seconds": seconds})
                print(f"[{count}/{total}] {row['id']} {variant} sample {sample}: "
                      f"{len(body.split())} words, {seconds:.0f}s", flush=True)
    print(f"Saved {out}")
    return 0


# ---- step 2: scoring (laptop) ------------------------------------------------------------------


def build_rows(frozen: list[dict], generations: list[dict]) -> list[dict]:
    """One row per report to score: every generation, plus the copy-the-notes baseline.

    An empty generation becomes a failed row, so a writer that produces nothing is counted as
    failing instead of being left out of its own average.
    """
    by_id = {r["id"]: r for r in frozen}
    rows = []
    for item in frozen:
        base = {k: item[k] for k in ("id", "topic", "question", "sources")}
        rows.append({**base, "variant": COPY, "sample": 0, "seconds": 0.0,
                     "report": make_report(copy_notes_writer(item), item["sources"]), "error": None})
    for gen in generations:
        item = by_id.get(gen["id"])
        if item is None:
            continue
        base = {k: item[k] for k in ("id", "topic", "question", "sources")}
        report = make_report(gen["body"], item["sources"]) if strip_thinking(gen["body"]) else ""
        rows.append({**base, "variant": gen["variant"], "sample": gen["sample"],
                     "seconds": gen.get("seconds", 0.0), "report": report,
                     "error": None if report else "the model wrote nothing"})
    return rows


def score_rows(
    rows: list[dict],
    judge,
    cache: Path,
    delay: float = 0.0,
    max_consecutive_failures: int = 3,
    on_progress=None,
) -> dict:
    """Score every row not already in `cache`, saving each result as soon as it exists.

    A report the judge could not rate at all (no faithfulness and no relevance) almost always means
    the provider refused the call, usually a spent daily quota. That is not saved, so a later run
    tries it again, and after a few in a row the run stops instead of wasting the rest of the quota.
    """
    cached = {key_of(r): r for r in read_lines(cache)}
    streak, stopped = 0, None
    pending = [r for r in rows if key_of(r) not in cached]
    for i, row in enumerate(pending, 1):
        if row["error"]:
            scored = {**row, "scores": {}}
        else:
            scored = score_row(row, judge)
            s = scored["scores"]
            if s.get("faithfulness") is None and s.get("relevance") is None:
                streak += 1
                if on_progress:
                    on_progress(i, len(pending), row, "judge gave no answer, will retry later")
                if streak >= max_consecutive_failures:
                    stopped = (f"stopped after {streak} reports the judge could not rate "
                               "(probably a spent daily quota); run the same command again later")
                    break
                continue
        streak = 0
        cached[key_of(scored)] = scored
        append_line(cache, scored)
        if on_progress:
            on_progress(i, len(pending), row, "scored")
        if delay and i < len(pending):
            time.sleep(delay)
    scored_rows = [cached[key_of(r)] for r in rows if key_of(r) in cached]
    return {"rows": scored_rows, "missing": len(rows) - len(scored_rows), "stopped": stopped}


def per_question(rows: list[dict], variant: str, metric: str) -> dict[str, float]:
    """Mean of a metric over a writer's samples, for each question that has a score."""
    values: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        score = r.get("scores", {}).get(metric)
        if r["variant"] == variant and not r.get("error") and score is not None:
            values[r["id"]].append(score)
    return {q: sum(v) / len(v) for q, v in values.items()}


def head_to_head(rows: list[dict], first: str, second: str, metric: str, margin: float = 0.005) -> dict:
    """On questions both writers have a score for: how often `first` beat, tied or lost to `second`."""
    a, b = per_question(rows, first, metric), per_question(rows, second, metric)
    shared = sorted(set(a) & set(b))
    wins = sum(a[q] > b[q] + margin for q in shared)
    losses = sum(a[q] < b[q] - margin for q in shared)
    mean = sum(a[q] - b[q] for q in shared) / len(shared) if shared else None
    return {"questions": len(shared), "wins": wins, "ties": len(shared) - wins - losses,
            "losses": losses, "mean_difference": mean}


def summarize(rows: list[dict]) -> dict:
    variants = [v for v in (COPY, *VARIANTS) if any(r["variant"] == v for r in rows)]
    return {
        "variants": {v: aggregate([r for r in rows if r["variant"] == v]) for v in variants},
        "tuned_vs_base": {key: head_to_head(rows, "tuned", "base", key)
                          for key in ("faithfulness", "relevance", "citation_coverage")},
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def to_markdown(result: dict) -> str:
    meta, summary = result["meta"], result["summary"]
    names = {"base": "Base model", "tuned": "Tuned (LoRA)", COPY: "Copy the notes"}
    variants = list(summary["variants"])
    header = (
        f"Base model: `{meta.get('base_model', '?')}` | Judge: `{meta.get('judge_model', '?')}` | "
        f"{meta.get('date', '')}"
    )
    lines = ["## Writer comparison", "", header, ""]
    if meta.get("stopped"):
        lines += [f"> **Incomplete:** {meta['stopped']}", ""]
    if meta.get("missing"):
        lines += [f"> {meta['missing']} reports are not scored yet.", ""]
    lines += ["| Metric | " + " | ".join(names[v] for v in variants) + " |",
              "|---|" + "---|" * len(variants)]
    for key, label, _ in METRICS:
        cells = [f"{_pct(summary['variants'][v].get(key))} (n={summary['variants'][v].get(f'{key}_n', 0)})"
                 for v in variants]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    counts = [f"{names[v]} {summary['variants'][v]['questions']} ({summary['variants'][v]['failed']} failed)"
              for v in variants]
    lines += ["", "Reports scored: " + "; ".join(counts) + ".", "",
              "Tuned against base, question by question (each question's score is the mean of its samples):",
              "", "| Metric | Questions | Tuned better | Tied | Tuned worse | Mean difference |",
              "|---|---|---|---|---|---|"]
    for key, row in summary["tuned_vs_base"].items():
        diff = "n/a" if row["mean_difference"] is None else f"{row['mean_difference'] * 100:+.1f} points"
        lines.append(f"| {key.replace('_', ' ').capitalize()} | {row['questions']} | {row['wins']} | "
                     f"{row['ties']} | {row['losses']} | {diff} |")
    return "\n".join(lines) + "\n"


def score(args: argparse.Namespace) -> int:
    from ..config import Settings
    from ..llm import make_llm
    from ..providers import check_llm

    frozen = load_frozen(Path(args.frozen))
    if args.n:
        frozen = frozen[: args.n]
    if not frozen:
        print(f"No frozen questions in {args.frozen}.", file=sys.stderr)
        return 2
    generations = read_lines(Path(args.generations))
    rows = build_rows(frozen, generations)
    if not generations:
        print("No generations found; only the copy-the-notes baseline will be scored.", file=sys.stderr)

    base = Settings.from_env()
    settings = replace(base, model=args.judge_model or base.model)
    error = check_llm(settings)
    if error:
        print(f"Cannot reach the judge ({settings.model}): {error}", file=sys.stderr)
        return 2
    judge = make_llm(settings, temperature=0.0)

    def progress(i: int, total: int, row: dict, status: str) -> None:
        print(f"[{i}/{total}] {row['id']} {row['variant']} sample {row['sample']}: {status}",
              file=sys.stderr, flush=True)

    out = Path(args.out_dir)
    outcome = score_rows(rows, judge, out / "scores.jsonl", delay=args.delay, on_progress=progress)
    meta = {"judge_model": settings.model, "date": date.today().isoformat(),
            "base_model": args.base_model, "missing": outcome["missing"], "stopped": outcome["stopped"]}
    result = {"meta": meta, "summary": summarize(outcome["rows"]), "rows": outcome["rows"]}
    (out / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "comparison.md").write_text(to_markdown(result), encoding="utf-8")
    print(to_markdown(result))
    print(f"Saved {out / 'comparison.json'} and {out / 'comparison.md'}", file=sys.stderr)
    return 1 if outcome["stopped"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scholarcrew-compare", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write reports with the base and tuned model (needs a GPU)")
    gen.add_argument("--adapter", default="outputs/writer-lora", help="folder with the trained adapter")
    gen.add_argument("--model", default="Qwen/Qwen3-4B", help="the base model the adapter was trained on")
    gen.add_argument("--frozen", default="data/frozen/eval", help="folder of frozen evaluation inputs")
    gen.add_argument("--out", default="generations.jsonl", help="where generated reports are saved")
    gen.add_argument("--samples", type=int, default=2, help="reports per writer and question")
    gen.add_argument("--max-new-tokens", type=int, default=1400)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--n", type=int, help="only the first N questions")

    sc = sub.add_parser("score", help="score the reports and compare the writers")
    sc.add_argument("--generations", default="generations.jsonl")
    sc.add_argument("--frozen", default="data/frozen/eval")
    sc.add_argument("--judge-model", help="judge model (default: LLM_MODEL from .env)")
    sc.add_argument("--base-model", default="Qwen/Qwen3-4B", help="only recorded in the results")
    sc.add_argument("--delay", type=float, default=3.0, help="seconds to pause between reports")
    sc.add_argument("--out-dir", default="data/compare", help="where scores and the table are saved")
    sc.add_argument("--n", type=int, help="only the first N questions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    return generate(args) if args.command == "generate" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
