"""Data handling for the LoRA training script. Needs no GPU and none of the training libraries."""

import pytest

from scholarcrew.finetune.train_lora import (
    build_parser,
    encode,
    encode_all,
    latest_checkpoint,
    split_holdout,
)


class FakeTokenizer:
    """One token per word. Records how the chat template was called."""

    eos_token = "<eos>"

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.template_calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        self.template_calls.append((tokenize, add_generation_prompt, enable_thinking))
        body = " ".join(f"<{m['role']}> {m['content']}" for m in messages)
        return body + " <assistant>"

    def __call__(self, text, add_special_tokens):
        words = text.replace("<eos>", " <eos>").split()  # the end token is its own token
        ids = [self.vocab.setdefault(word, len(self.vocab)) for word in words]
        return {"input_ids": ids}


def example(i=0, answer="a grounded report with a cited claim [1]"):
    return {"id": f"tr-{i:03d}", "messages": [
        {"role": "system", "content": "write the report"},
        {"role": "user", "content": "Question: q Notes: n [1]"},
        {"role": "assistant", "content": answer},
    ]}


def test_holdout_is_deterministic_disjoint_and_sized():
    rows = [example(i) for i in range(100)]
    train, val = split_holdout(rows, 0.1, seed=0)
    assert len(val) == 10 and len(train) == 90
    assert not {r["id"] for r in train} & {r["id"] for r in val}
    assert split_holdout(rows, 0.1, seed=0) == (train, val)  # repeatable
    assert split_holdout(rows, 0.1, seed=1)[1] != val  # another seed, another split


def test_holdout_edge_cases():
    rows = [example(i) for i in range(5)]
    assert split_holdout(rows, 0.0, seed=0)[1] == []  # none asked for
    assert split_holdout(rows, 0.1, seed=0)[1] == []  # too few examples to spare any
    assert len(split_holdout([example(i) for i in range(12)], 0.01, seed=0)[1]) == 1  # at least one
    train, val = split_holdout([example(0), example(1)], 1.0, seed=0)
    assert len(train) == 1 and len(val) == 1  # never an empty training set


def test_loss_is_counted_on_the_report_only_and_ends_with_the_end_token():
    tok = FakeTokenizer()
    out = encode(example(), tok, max_len=100)
    prompt = tok.apply_chat_template(example()["messages"][:2], False, True, False)
    prompt_len = len(tok(prompt, False)["input_ids"])
    assert out["labels"][:prompt_len] == [-100] * prompt_len  # the prompt is not trained on
    assert out["labels"][prompt_len:] == out["input_ids"][prompt_len:]  # the report is
    assert out["input_ids"][-1] == tok.vocab["<eos>"]
    assert len(out["labels"]) == len(out["input_ids"])


def test_prompt_is_built_the_way_inference_builds_it():
    tok = FakeTokenizer()
    encode(example(), tok, max_len=100)
    # not tokenized by the template, assistant turn opened, thinking off
    assert tok.template_calls[0] == (False, True, False)


def test_examples_that_do_not_fit_are_skipped_not_truncated():
    tok = FakeTokenizer()
    assert encode(example(), tok, max_len=5) is None
    kept, skipped = encode_all([example(0), example(1, answer="x " * 500)], tok, max_len=60)
    assert len(kept) == 1 and skipped == 1


def test_collator_pads_labels_with_ignore_and_masks_attention():
    pytest.importorskip("torch")
    from scholarcrew.finetune.train_lora import make_collator

    batch = make_collator(pad_id=9)([
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "labels": [-100, 5]},
    ])
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 9]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_defaults_fit_a_16gb_gpu():
    """Regression: batch 2 at 4096 tokens ran out of memory on a free T4. The 151k-word vocabulary
    makes the output layer large, so one example at a time, with updates every 16, is the safe setting."""
    args = build_parser().parse_args([])
    assert args.model == "Qwen/Qwen3-8B" and args.four_bit and args.max_len == 3072
    assert args.batch_size == 1 and args.batch_size * args.grad_accum == 16
    assert build_parser().parse_args(["--no-4bit"]).four_bit is False


def test_loader_accepts_a_file_saved_with_a_windows_byte_order_mark(tmp_path):
    import json

    from scholarcrew.finetune.train_lora import load_examples

    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(example(1)) + "\n" + json.dumps(example(2)) + "\n", encoding="utf-8-sig")
    assert [r["id"] for r in load_examples(path)] == ["tr-001", "tr-002"]


def test_resume_finds_the_newest_checkpoint_by_step_number(tmp_path):
    assert latest_checkpoint(tmp_path) is None  # nothing saved yet
    assert latest_checkpoint(tmp_path / "missing") is None
    for name in ("checkpoint-3", "checkpoint-12", "checkpoint-6"):
        (tmp_path / name).mkdir()
    (tmp_path / "checkpoint-99.txt").write_text("not a folder")
    (tmp_path / "notes").mkdir()
    assert latest_checkpoint(tmp_path).endswith("checkpoint-12")  # 12 beats 6 and 3, not text order


def test_the_held_out_check_uses_the_same_small_batch_as_training():
    """Regression: the evaluation batch defaulted to 8, and 8 long examples at once overflowed a
    16 GB GPU (a 5.27 GiB allocation) when the first held-out check ran."""
    import inspect

    from scholarcrew.finetune import train_lora

    source = inspect.getsource(train_lora.train)
    assert '"per_device_eval_batch_size": args.batch_size' in source


def test_training_defaults_are_crash_tolerant():
    args = build_parser().parse_args([])
    assert args.save_every == 3 and not args.fresh and not args.progress_bar
