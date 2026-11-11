"""Freezing the writer's inputs, and keeping the training and evaluation questions apart. Offline."""

import json

import pytest

from conftest import FakeLLM, fake_arxiv, fake_web
from scholarcrew.evaluation.runner import LLMFailure, load_questions
from scholarcrew.finetune.freeze import (
    freeze,
    freeze_one,
    load_frozen,
    load_train_questions,
    overlap,
)

ITEM = {"id": "tr-001", "topic": "RAG", "question": "How does X work?"}


def test_training_and_evaluation_questions_are_disjoint():
    train = load_train_questions()
    assert len(train) >= 100
    assert len({q["id"] for q in train}) == len(train)
    assert overlap(train, load_questions()) == []
    assert overlap([{"id": "t", "question": "How does LoRA fine-tuning compare with full fine-tuning"}],
                   load_questions()) == ["t"]


def test_freeze_one_keeps_sources_and_notes():
    row = freeze_one(ITEM, FakeLLM(), fake_arxiv, fake_web)
    assert row["id"] == "tr-001" and len(row["sources"]) == 2
    assert "[1]" in row["notes"] and "report" not in row


def test_freeze_one_refuses_a_degraded_result():
    with pytest.raises(LLMFailure):
        freeze_one(ITEM, FakeLLM(fail=True), fake_arxiv, fake_web)
    with pytest.raises(LLMFailure, match="no sources"):
        freeze_one(ITEM, FakeLLM(), lambda q, n: [], lambda q, n: [])


def test_freeze_resumes_and_skips_saved_questions(tmp_path):
    items = [{**ITEM, "id": f"tr-00{i}"} for i in (1, 2, 3)]
    (tmp_path / "tr-001.json").write_text("{}", encoding="utf-8")
    seen = []

    def work(item):
        seen.append(item["id"])
        return {**item, "sources": [], "notes": "n"}

    counts = freeze(items, tmp_path, work)
    assert seen == ["tr-002", "tr-003"] and counts["skipped"] == 1 and counts["saved"] == 2
    assert json.loads((tmp_path / "tr-002.json").read_text("utf-8"))["notes"] == "n"
    assert len(load_frozen(tmp_path)) == 3


def test_freeze_stops_after_repeated_failures(tmp_path):
    items = [{**ITEM, "id": f"tr-00{i}"} for i in range(1, 8)]
    calls = []

    def broken(item):
        calls.append(item["id"])
        raise LLMFailure("rate limited")

    counts = freeze(items, tmp_path, broken, max_consecutive_failures=3)
    assert counts["aborted"] and counts["failed"] == 3 and len(calls) == 3
    assert not list(tmp_path.glob("*.json"))  # nothing half-written
