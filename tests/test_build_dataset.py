"""Building the fine-tuning set: only grounded reports are kept. Offline."""

import json

from langchain_core.messages import AIMessage

from scholarcrew.agents.writer import SYSTEM, writer_input
from scholarcrew.finetune.build_dataset import build, draft, make_example, quality_problems

SOURCES = [
    {"id": 1, "kind": "arxiv", "title": "Late chunking", "url": "u1", "cite": "c1",
     "text": "Late chunking improved faithfulness by 13% over fixed-size chunks."},
    {"id": 2, "kind": "web", "title": "Guide", "url": "u2", "cite": "c2",
     "text": "A practical guide to chunking with no reported results."},
]
FILLER = " ".join(["Careful chunking choices change what the retriever can find for the model [2]."] * 22)
GOOD = (
    "# Chunking\n\n## Summary\n\nLate chunking improved faithfulness by 13% over fixed-size chunks [1].\n\n"
    f"## Details\n\n{FILLER}\n\n## Limitations\n\nThe evidence comes from a single source and is thin [1].\n"
)
ROW = {"id": "tr-001", "question": "q?", "notes": "n [1]", "sources": SOURCES}


class Script:
    def __init__(self, reply=None, fail=False):
        self.reply, self.fail = reply, fail

    def invoke(self, _messages):
        if self.fail:
            raise RuntimeError("rate limited")
        return AIMessage(content=self.reply)


def test_a_grounded_report_is_kept():
    assert quality_problems(GOOD, SOURCES) == []


def test_each_kind_of_bad_report_is_rejected():
    invented = GOOD.replace("13%", "47%")
    assert "states a figure its cited source does not contain" in quality_problems(invented, SOURCES)
    bad_cite = GOOD.replace("[1].\n\n## Details", "[9].\n\n## Details")
    assert "cites a source that does not exist" in quality_problems(bad_cite, SOURCES)
    uncited = GOOD.replace(" [2]", "")
    assert any("cited" in p for p in quality_problems(uncited, SOURCES))
    assert "missing title, sections or length" in quality_problems("# T\n\nShort [1].", SOURCES)
    assert quality_problems("   ", SOURCES) == ["empty"]


def test_example_uses_the_writers_own_prompt_and_input_format():
    messages = make_example(ROW, GOOD)["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == SYSTEM
    assert messages[1]["content"] == writer_input("q?", "n [1]")
    assert messages[2]["content"] == GOOD


def test_draft_normalizes_brackets_and_surfaces_model_failures():
    assert draft(Script("a【1】"), ROW) == "a[1]"
    try:
        draft(Script(fail=True), ROW)
    except RuntimeError as exc:
        assert "rate limited" in str(exc)
    else:
        raise AssertionError("a failed model call must not look like an empty report")


def test_build_keeps_logs_rejects_and_resumes(tmp_path):
    rows = [{**ROW, "id": f"tr-00{i}"} for i in (1, 2)]
    replies = {"tr-001": GOOD, "tr-002": "# T\n\nShort [1]."}
    counts = build(rows, tmp_path, lambda r: replies[r["id"]])
    assert counts["kept"] == 1 and counts["rejected"] == 1
    kept = [json.loads(x) for x in (tmp_path / "train.jsonl").read_text("utf-8").splitlines()]
    assert [k["id"] for k in kept] == ["tr-001"]
    assert "tr-002" in (tmp_path / "rejected.jsonl").read_text("utf-8")

    def must_not_run(_row):
        raise AssertionError("an already handled question was drafted again")

    assert build(rows, tmp_path, must_not_run)["skipped"] == 2


def test_build_stops_after_repeated_failures(tmp_path):
    rows = [{**ROW, "id": f"tr-00{i}"} for i in range(1, 8)]
    calls = []

    def broken(row):
        calls.append(row["id"])
        raise RuntimeError("429")

    counts = build(rows, tmp_path, broken)
    assert counts["aborted"] and len(calls) == 3 and not (tmp_path / "train.jsonl").exists()


def test_copy_notes_baseline_returns_the_notes_with_a_title():
    from scholarcrew.finetune.baselines import copy_notes_writer

    body = copy_notes_writer({"question": "q?", "notes": "A finding【1】."})
    assert body == "# q?\n\nA finding[1]."
    assert copy_notes_writer({"question": "q?", "notes": "# Own title\n\nx [1]"}).startswith("# Own title")
