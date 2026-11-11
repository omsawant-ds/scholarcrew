"""Offline tests for the evaluation pipeline: no network, no API key."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from conftest import FakeLLM, fake_arxiv, fake_web
from scholarcrew.evaluation import __main__ as cli
from scholarcrew.evaluation import (
    aggregate,
    check_thresholds,
    deterministic_scores,
    extract_claims,
    load_questions,
    run,
    runner,
    to_markdown,
)
from scholarcrew.evaluation import judge as judge_mod
from scholarcrew.evaluation.judge import judge_faithfulness, judge_relevance
from scholarcrew.evaluation.metrics import (
    citation_scores,
    numeric_grounding,
    structure_score,
)
from scholarcrew.grounding import cite_ids, split_report, statistics_in

REPORT = """# Title of the report

**Summary** Grounding retrieved passages in the answer reduces unsupported statements in output [1].

## First section

Rerankers reorder candidate passages so that the most relevant evidence comes first in context [2][3].
Short line.

| Method | Effect |
|---|---|
| Hybrid search | Combining sparse and dense retrieval improves recall on rare terms [1] |

## Second section

This sentence makes a claim but cites a source that does not exist in the list at all [9].
This sentence has no citation at all but is long enough to count as a substantive statement.

## References

[1] A. Author (2024). Paper one. http://arxiv.org/abs/1
[2] B. Author (2023). Paper two. http://arxiv.org/abs/2
[3] Blog. https://example.com
"""


class Scripted:
    """A judge that returns canned replies in order, then repeats the last one."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def invoke(self, _messages):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply)


class EvalFake(FakeLLM):
    """Serves the crew's prompts and the judge's prompts from one object."""

    def invoke(self, messages):
        system = messages[0].content
        if "fact-checker" in system:
            return AIMessage(content=json.dumps(
                {"verdicts": [{"id": i, "supported": i % 2 == 1} for i in range(1, 13)]}))
        if "grading a research report" in system:
            return AIMessage(content='{"score": 5, "reason": "good"}')
        return super().invoke(messages)


SOURCES = [{"id": i, "kind": "arxiv", "title": f"T{i}", "url": f"u{i}", "text": "x" * 50, "cite": "c"}
           for i in (1, 2, 3)]


# ---------------------------------------------------------------- metrics

def test_split_report_separates_references():
    body, refs = split_report(REPORT)
    assert "Second section" in body and "## References" not in body
    assert refs.startswith("## References") and "[3] Blog" in refs


def test_extract_claims_finds_cited_statements_with_their_ids():
    claims = extract_claims(REPORT)
    cited = [c["cited"] for c in claims]
    reranker = next(c for c in claims if c["text"].startswith("Rerankers reorder"))
    assert reranker["cited"] == [2, 3]
    assert [1] in cited and [9] in cited
    assert all(c["text"] and "[" not in c["text"] for c in claims)
    assert not any("Short line" in c["text"] for c in claims)  # too short to be a claim


def test_extract_claims_is_capped_and_evenly_sampled():
    many = "\n".join(f"Statement number {i} says something specific about retrieval systems [1]." for i in range(40))
    claims = extract_claims(many, cap=10)
    assert len(claims) == 10
    assert "number 0 " in claims[0]["text"] and "number 36 " in claims[-1]["text"]


def test_cite_ids_expands_ranges_lists_and_adjacent_groups():
    text = "a [1] b [2][3] c [4, 6] d [18-20] e [21‑22] f [x] g []"  # U+2011 non-breaking hyphen
    assert cite_ids(text) == [1, 2, 3, 4, 6, 18, 19, 20, 21, 22]
    assert cite_ids("no citations here") == []


def test_table_row_is_one_statement_so_claim_and_citation_stay_together():
    # Regression: the claim and its citation sit in different cells of the same row.
    table = (
        "| Technique | Evidence |\n|---|---|\n"
        "| **Iterative refinement** fills missing evidence gaps before the final answer | FAIR-RAG [1]; ReaRAG [6] |\n"
    )
    (claim,) = extract_claims(table)
    assert claim["cited"] == [1, 6]
    assert "Iterative refinement fills" in claim["text"] and "**" not in claim["text"]
    assert "Technique" not in claim["text"]  # header row skipped
    assert citation_scores(table, 6)["citation_coverage"] == 1.0


def test_numeric_grounding_catches_invented_statistics():
    sources = [
        {"id": 1, "title": "Late chunking", "text": "Faithfulness improved by 13% over fixed-size chunks.",
         "url": "u", "kind": "arxiv", "cite": "c"},
        {"id": 2, "title": "Other", "text": "Recall reached 0.71 on the benchmark.", "url": "u2",
         "kind": "web", "cite": "c"},
    ]
    report = (
        "# T\n\n"
        "Late chunking raised faithfulness by 13 % over fixed-size chunks [1].\n"        # grounded
        "Faithfulness rose from 0.71 to 0.84 in the same experiment overall [1].\n"      # 0.84 invented, 0.71 in other source
        "Adaptive selection gave a 9% boost across all of the tested corpora here.\n"    # no citation at all
        "## References\n[1] x\n"
    )
    # statistics: 13 (ok), 0.71 (not in source 1), 0.84 (absent), 9 (uncited) -> 1 of 4
    assert numeric_grounding(report, sources) == pytest.approx(1 / 4)


def test_numeric_grounding_ignores_years_list_numbers_and_model_names():
    sources = [{"id": 1, "title": "t", "text": "no figures here", "url": "u", "kind": "web", "cite": "c"}]
    report = "# T\n\nIn 2024 the GPT-3.5 and Llama‑3.1 models were compared in 3 experiments [1].\n"
    assert statistics_in(report) == []
    assert numeric_grounding(report, sources) is None  # nothing to check is not a perfect score


def test_citation_ranges_count_as_valid_sources():
    s = citation_scores("# T\n\nPractical guides cover this topic in considerable detail across sources [3-5].", 5)
    assert s["citation_validity"] == 1.0 and s["source_utilization"] == pytest.approx(3 / 5)
    assert citation_scores("# T\n\nThis claim cites a range that runs past the sources list [4-8] entirely.", 5)[
        "citation_validity"] == pytest.approx(2 / 5)


def test_citation_scores_detect_invalid_and_missing_citations():
    s = citation_scores(REPORT, n_sources=3)
    assert s["citation_validity"] == pytest.approx(4 / 5)  # [9] is invalid
    assert 0 < s["citation_coverage"] < 1                 # one substantive sentence is uncited
    assert s["source_utilization"] == pytest.approx(1.0)  # sources 1, 2, 3 all cited


def test_structure_score_and_empty_report():
    assert structure_score(REPORT) == pytest.approx(0.75)  # all but the 250-word length check
    assert structure_score("no headings, no references") == 0.0
    assert deterministic_scores("   ", 3) == {
        "citation_validity": 0.0, "citation_coverage": 0.0, "source_utilization": 0.0, "structure": 0.0}


# ---------------------------------------------------------------- judge

def test_faithfulness_is_the_share_of_supported_claims():
    claims = [{"text": f"claim {i} is long enough", "cited": [1]} for i in range(4)]
    judge = Scripted('{"verdicts": [{"id":1,"supported":true},{"id":2,"supported":true},'
                     '{"id":3,"supported":true},{"id":4,"supported":false}]}')
    assert judge_faithfulness(judge, claims, SOURCES) == 0.75


@pytest.fixture(autouse=True)
def no_real_waiting(monkeypatch):
    """The judge waits between retries; tests must not."""
    monkeypatch.setattr(judge_mod, "_pause", lambda seconds: None)


def test_judge_waits_between_retries_so_rate_limits_can_clear(monkeypatch):
    waits = []
    monkeypatch.setattr(judge_mod, "_pause", waits.append)
    judge = Scripted("", "", '{"score": 4}')  # two empty replies (rate limited), then an answer
    assert judge_relevance(judge, "q", "r") == 0.75
    assert waits == [20.0, 20.0] and judge.calls == 3


def test_judge_retries_once_on_garbage_then_succeeds():
    claims = [{"text": "a claim that is long enough", "cited": [1]}]
    judge = Scripted("sorry, I cannot do that", '```json\n{"verdicts": [{"id": 1, "supported": true}]}\n```')
    assert judge_faithfulness(judge, claims, SOURCES) == 1.0
    assert judge.calls == 2


@pytest.mark.parametrize("reply", ["", "not json", '{"verdicts": "nope"}', '{"verdicts": []}'])
def test_unusable_judge_output_gives_none_not_zero(reply):
    claims = [{"text": "a claim that is long enough", "cited": [1]}]
    assert judge_faithfulness(Scripted(reply), claims, SOURCES) is None


def test_judge_errors_give_none_and_no_claims_gives_none():
    claims = [{"text": "a claim that is long enough", "cited": [1]}]
    assert judge_faithfulness(Scripted(RuntimeError("429")), claims, SOURCES) is None
    assert judge_faithfulness(Scripted("{}"), [], SOURCES) is None


@pytest.mark.parametrize(("reply", "expected"), [
    ('{"score": 5}', 1.0), ('{"score": 3, "reason": "ok"}', 0.5), ('{"score": 1}', 0.0),
    ('{"score": 7}', None), ('{"score": 0}', None), ('{"score": true}', None),
    ('{"score": "high"}', None), ("garbage", None),
])
def test_relevance_rescales_and_validates(reply, expected):
    assert judge_relevance(Scripted(reply), "q", "report") == expected


# ---------------------------------------------------------------- runner

def fake_generate(question):
    return {"report": REPORT, "sources": SOURCES, "log": ["writer: done"], "seconds": 2.0}


ITEMS = [{"id": "a-1", "topic": "A", "question": "q1?"}, {"id": "b-1", "topic": "B", "question": "q2?"}]


def test_run_scores_every_question_and_aggregates():
    result = run(ITEMS, fake_generate, EvalFake(), meta={"model": "m", "judge_model": "j"})
    s = result["summary"]
    assert s["questions"] == 2 and s["failed"] == 0
    assert s["relevance"] == 1.0 and 0 <= s["faithfulness"] <= 1
    assert s["citation_validity"] == pytest.approx(0.8)
    assert s["mean_seconds"] == 2.0 and s["mean_sources"] == 3.0


def test_a_failing_question_is_recorded_and_excluded_from_means():
    def flaky(question):
        if question == "q1?":
            raise RuntimeError("provider down")
        return fake_generate(question)

    result = run(ITEMS, flaky, EvalFake())
    assert result["summary"]["failed"] == 1 and result["summary"]["questions"] == 2
    assert "provider down" in result["rows"][0]["error"]
    assert result["summary"]["relevance_n"] == 1  # only the good row counts


def test_degraded_reports_from_failed_model_calls_are_not_scored(tmp_path):
    """Regression: a rate-limited run used to be scored as if the system had worked."""

    class RateLimited(EvalFake):
        def invoke(self, messages):
            raise RuntimeError("429 tokens per day exhausted")

    gen = runner.make_generator(RateLimited(), runner.Settings("http://x", "k", "m"),
                                arxiv_search=fake_arxiv, web_search=fake_web)
    with pytest.raises(runner.LLMFailure, match="model call.*failed"):
        gen("q?")
    result = run(ITEMS[:1], gen, EvalFake())
    assert result["summary"]["failed"] == 1 and result["summary"]["faithfulness"] is None
    assert "429" in result["rows"][0]["error"]


def test_run_aborts_after_repeated_failures_to_save_quota():
    calls = []

    def always_fails(question):
        calls.append(question)
        raise runner.LLMFailure("provider down")

    items = [{"id": f"q{i}", "topic": "T", "question": f"q{i}?"} for i in range(8)]
    result = run(items, always_fails, EvalFake(), max_consecutive_failures=3)
    assert len(calls) == 3 and len(result["rows"]) == 3
    assert "stopped after 3 failures" in result["meta"]["aborted"]
    assert "Run aborted" in to_markdown(result)


def test_a_success_resets_the_failure_streak():
    outcomes = iter([False, False, True, False, False, True])

    def flaky(question):
        if not next(outcomes):
            raise runner.LLMFailure("blip")
        return fake_generate(question)

    items = [{"id": f"q{i}", "topic": "T", "question": f"q{i}?"} for i in range(6)]
    result = run(items, flaky, EvalFake(), max_consecutive_failures=3)
    assert len(result["rows"]) == 6 and "aborted" not in result["meta"]


def test_report_without_any_citation_scores_zero_faithfulness_not_missing():
    """A report that cites nothing must not escape the average as 'no data'."""
    row = {"id": "x", "topic": "T", "question": "q?", "error": None, "seconds": 1.0, "log": [],
           "report": "# Title\n\nA long paragraph with many words but absolutely no citation markers "
                     "anywhere in it at all.\n\n## References\n[1] x\n",
           "sources": SOURCES}
    scored = runner.score_row(row, EvalFake())
    assert scored["scores"]["faithfulness"] == 0.0 and scored["scores"]["citation_validity"] == 0.0


def test_resume_merge_keeps_successes_and_replaces_failures():
    meta = {"model": "m", "judge_model": "j"}
    items = [{"id": "rag-01", "topic": "RAG", "question": "q1?"},
             {"id": "rag-02", "topic": "RAG", "question": "q2?"}]

    def flaky(question):
        if question == "q2?":
            raise RuntimeError("429 daily limit")
        return fake_generate(question)

    first = run(items, flaky, EvalFake(), meta=meta)
    assert first["summary"]["failed"] == 1
    again = run(items[1:], fake_generate, EvalFake(), meta=meta)
    merged = runner.resume_merge(first, again)
    assert [r["id"] for r in merged["rows"]] == ["rag-01", "rag-02"]
    assert merged["summary"]["failed"] == 0 and merged["summary"]["questions"] == 2
    assert merged["meta"]["resumed"] is True


def test_judge_failures_are_counted_not_averaged_as_zero():
    result = run(ITEMS[:1], fake_generate, Scripted(RuntimeError("rate limited")))
    s = result["summary"]
    assert s["faithfulness"] is None and s["faithfulness_n"] == 0
    assert s["citation_validity"] is not None  # deterministic metrics still work


def test_aggregate_handles_no_successful_rows():
    s = aggregate([{"error": "x", "scores": {}, "sources": [], "seconds": 0}])
    assert s["failed"] == 1 and s["faithfulness"] is None and s["mean_seconds"] is None


def test_thresholds():
    summary = {"faithfulness": 0.9, "relevance": 0.5, "citation_validity": None}
    assert check_thresholds(summary, {"faithfulness": 0.8}) == []
    msgs = check_thresholds(summary, {"relevance": 0.7, "citation_validity": 0.9})
    assert len(msgs) == 2 and "below" in msgs[0] and "no score" in msgs[1]


def test_rescore_changes_judge_scores_without_regenerating():
    first = run(ITEMS, fake_generate, EvalFake())
    again = runner.rescore(first, Scripted('{"score": 1}'), meta={"judge_model": "other"})
    assert again["summary"]["relevance"] == 0.0 and first["summary"]["relevance"] == 1.0
    assert again["meta"]["judge_model"] == "other"
    assert again["rows"][0]["report"] == first["rows"][0]["report"]


def test_markdown_and_saved_files(tmp_path):
    result = run(ITEMS, fake_generate, EvalFake(), meta={"model": "m", "judge_model": "j", "date": "2026-01-01"})
    md = to_markdown(result)
    assert "| Faithfulness |" in md and "Generator: `m`" in md and "a-1" in md
    json_path, md_path = runner.save(result, tmp_path / "out")
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["questions"] == 2
    assert md_path.read_text(encoding="utf-8").startswith("## Evaluation results")


def test_question_set_is_well_formed_and_filterable():
    questions = load_questions()
    assert len(questions) >= 25 and len({q["id"] for q in questions}) == len(questions)
    assert all(q["question"].endswith("?") and q["topic"] for q in questions)
    assert len(load_questions(n=3)) == 3
    rag = load_questions(topic="rag")
    assert rag and all(q["topic"] == "RAG" for q in rag)


# ---------------------------------------------------------------- CLI

@pytest.fixture
def offline_cli(monkeypatch):
    real = runner.stream_crew
    monkeypatch.setattr(cli, "make_llm", lambda s, temperature=0.2: EvalFake())
    monkeypatch.setattr(cli, "check_llm", lambda s: None)
    monkeypatch.setattr(
        runner, "stream_crew",
        lambda q, m, s, **kw: real(q, m, s, arxiv_search=fake_arxiv, web_search=fake_web))


def test_cli_end_to_end(offline_cli, tmp_path, capsys):
    code = cli.main(["--n", "2", "--delay", "0", "--out-dir", str(tmp_path)])
    out = capsys.readouterr()
    assert code == 0
    assert "## Evaluation results" in out.out and "[2/2]" in out.err
    assert len(list(Path(tmp_path).glob("eval-*.json"))) == 1


def test_cli_threshold_failure_sets_exit_code(offline_cli, tmp_path, capsys):
    code = cli.main(["--n", "1", "--delay", "0", "--out-dir", str(tmp_path), "--min-faithfulness", "1.01"])
    assert code == 1 and "THRESHOLD FAILED" in capsys.readouterr().err


def test_cli_resume_reruns_only_the_failed_questions(offline_cli, tmp_path, capsys):
    items = load_questions(n=2)

    def flaky(question):
        if question == items[1]["question"]:
            raise RuntimeError("429")
        return fake_generate(question)

    first = run(items, flaky, EvalFake(), meta={"model": "m1", "judge_model": "j1"})
    path, _ = runner.save(first, tmp_path)
    code = cli.main(["--n", "2", "--delay", "0", "--model", "m1", "--judge-model", "j1",
                     "--resume", str(path), "--out-dir", str(tmp_path / "out")])
    assert code == 0
    assert "[1/1]" in capsys.readouterr().err  # only the one failed question ran again
    merged = json.loads(next((tmp_path / "out").glob("eval-*.json")).read_text(encoding="utf-8"))
    assert merged["summary"]["questions"] == 2 and merged["summary"]["failed"] == 0


def test_cli_refuses_to_resume_with_a_different_model(offline_cli, tmp_path, capsys):
    first = run(load_questions(n=1), fake_generate, EvalFake(), meta={"model": "m1", "judge_model": "j1"})
    path, _ = runner.save(first, tmp_path)
    code = cli.main(["--n", "1", "--model", "other", "--judge-model", "j1", "--resume", str(path)])
    assert code == 2 and "Mixing models" in capsys.readouterr().err


def test_cli_stops_early_on_a_bad_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "make_llm", lambda s, temperature=0.2: EvalFake())
    monkeypatch.setattr(cli, "check_llm", lambda s: "AuthenticationError: bad key")
    assert cli.main(["--n", "1", "--out-dir", str(tmp_path)]) == 2
    assert "Cannot reach the generator model" in capsys.readouterr().err
