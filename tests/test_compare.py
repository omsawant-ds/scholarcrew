"""Comparing writers: building the reports to score, scoring with resume, and the table. Offline."""

import json

import pytest
from langchain_core.messages import AIMessage

from scholarcrew.evaluation import judge as judge_module
from scholarcrew.finetune.compare import (
    COPY,
    build_parser,
    build_rows,
    head_to_head,
    make_report,
    read_lines,
    score_rows,
    strip_thinking,
    summarize,
    to_markdown,
)

SOURCES = [
    {"id": 1, "kind": "arxiv", "title": "Late chunking", "url": "u1", "cite": "c1",
     "text": "Late chunking improved faithfulness by 13% over fixed-size chunks."},
    {"id": 2, "kind": "web", "title": "Guide", "url": "u2", "cite": "c2",
     "text": "A practical guide to chunking with no reported results."},
]
BODY = "# Chunking\n\n## Summary\n\nLate chunking improved faithfulness by 13% over fixed-size chunks [1].\n"
FROZEN = [
    {"id": "ev-01", "topic": "RAG", "question": "q1?", "sources": SOURCES, "notes": "Late chunking helps [1]."},
    {"id": "ev-02", "topic": "RAG", "question": "q2?", "sources": SOURCES, "notes": "Guides exist [2]."},
]


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(judge_module, "_pause", lambda _seconds: None)  # the judge waits between retries


def gen(id_, variant, sample=0, body=BODY):
    return {"id": id_, "variant": variant, "sample": sample, "body": body, "seconds": 5.0}


class Judge:
    """Answers the faithfulness and relevance questions, or fails like a spent quota."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def invoke(self, messages):
        self.calls += 1
        if self.fail:
            raise RuntimeError("rate limited")
        if "fact-checker" in messages[0].content:
            return AIMessage(content='{"verdicts": [{"id": 1, "supported": true}]}')
        return AIMessage(content='{"score": 4, "reason": "ok"}')


def test_thinking_blocks_are_removed_finished_or_cut_off():
    assert strip_thinking("<think>plan</think>\n# Title") == "# Title"
    assert strip_thinking("# Title\n<think>never finished") == "# Title"
    assert strip_thinking("# Title") == "# Title"


def test_a_generated_body_gets_the_real_reference_list_and_plain_citations():
    report = make_report("# T\n\nA claim 【1】.", SOURCES)
    assert "A claim [1]." in report
    assert report.rstrip().endswith("[2] c2") and "## References" in report


def test_rows_cover_every_generation_and_the_copy_baseline():
    rows = build_rows(FROZEN, [gen("ev-01", "base"), gen("ev-01", "tuned"), gen("gone", "base")])
    kinds = sorted((r["id"], r["variant"]) for r in rows)
    assert kinds == [("ev-01", "base"), ("ev-01", COPY), ("ev-01", "tuned"), ("ev-02", COPY)]  # unknown id dropped
    copy = next(r for r in rows if r["variant"] == COPY and r["id"] == "ev-01")
    assert "Late chunking helps [1]." in copy["report"] and "## References" in copy["report"]


def test_an_empty_generation_is_a_failure_not_a_missing_row():
    rows = build_rows(FROZEN, [gen("ev-01", "base", body="<think>only thinking</think>")])
    row = next(r for r in rows if r["variant"] == "base")
    assert row["error"] == "the model wrote nothing" and row["report"] == ""


def test_scoring_saves_each_result_and_does_not_repeat_finished_work(tmp_path):
    rows = build_rows(FROZEN, [gen("ev-01", "base"), gen("ev-01", "tuned")])
    cache, judge = tmp_path / "scores.jsonl", Judge()
    first = score_rows(rows, judge, cache)
    assert first["missing"] == 0 and len(first["rows"]) == len(rows)
    calls = judge.calls
    again = score_rows(rows, judge, cache)
    assert judge.calls == calls  # nothing was judged a second time
    assert len(again["rows"]) == len(rows) and len(read_lines(cache)) == len(rows)


def test_reports_the_judge_could_not_rate_are_retried_later_and_the_run_stops(tmp_path):
    rows = build_rows(FROZEN, [gen("ev-01", "base"), gen("ev-01", "tuned"), gen("ev-02", "base")])
    cache = tmp_path / "scores.jsonl"
    outcome = score_rows(rows, Judge(fail=True), cache, max_consecutive_failures=3)
    assert outcome["stopped"] and "daily quota" in outcome["stopped"]
    # A refused call is never saved as a score. (The two copy-the-notes reports have no cited claims, so
    # they score 0% faithfulness without asking the judge, and that is a real result.)
    assert {r["variant"] for r in read_lines(cache)} <= {COPY}
    assert outcome["missing"] == 3  # the three generated reports are still waiting
    later = score_rows(rows, Judge(), cache)  # the quota is back
    assert later["missing"] == 0 and not later["stopped"]


def scored(variant, id_, faith, sample=0, error=None):
    return {"id": id_, "variant": variant, "sample": sample, "error": error, "seconds": 1.0, "sources": SOURCES,
            "scores": {} if error else {"faithfulness": faith, "relevance": 0.75}}


def test_head_to_head_counts_wins_ties_and_losses_per_question():
    rows = [scored("tuned", "a", 0.9), scored("base", "a", 0.6),
            scored("tuned", "b", 0.5), scored("base", "b", 0.5),
            scored("tuned", "c", 0.4), scored("base", "c", 0.8),
            scored("tuned", "d", 1.0), scored("base", "d", None, error="the model wrote nothing")]
    result = head_to_head(rows, "tuned", "base", "faithfulness")
    assert (result["questions"], result["wins"], result["ties"], result["losses"]) == (3, 1, 1, 1)
    assert abs(result["mean_difference"] - (0.3 + 0.0 - 0.4) / 3) < 1e-9  # question d has no base score


def test_samples_are_averaged_within_a_question_before_comparing():
    rows = [scored("tuned", "a", 1.0, sample=0), scored("tuned", "a", 0.0, sample=1), scored("base", "a", 0.6)]
    assert head_to_head(rows, "tuned", "base", "faithfulness")["losses"] == 1  # 0.5 average vs 0.6


def test_the_table_lists_each_writer_and_says_when_it_is_incomplete():
    rows = [{**scored(v, "a", f), "scores": {"faithfulness": f, "relevance": 0.75}}
            for v, f in ((COPY, 1.0), ("base", 0.6), ("tuned", 0.9))]
    result = {"meta": {"judge_model": "j", "base_model": "m", "date": "2026-11-14", "stopped": "quota", "missing": 4},
              "summary": summarize(rows), "rows": rows}
    text = to_markdown(result)
    assert "| Metric | Copy the notes | Base model | Tuned (LoRA) |" in text
    assert "| Faithfulness | 100% (n=1) | 60% (n=1) | 90% (n=1) |" in text
    assert "Incomplete" in text and "4 reports are not scored yet" in text
    assert "+30.0 points" in text


def test_defaults_match_what_training_used():
    args = build_parser().parse_args(["generate"])
    assert args.model == "Qwen/Qwen3-4B" and args.samples == 2
    assert json.loads(json.dumps(build_parser().parse_args(["score"]).out_dir)) == "data/compare"
