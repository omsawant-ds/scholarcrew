"""Fine-tune a small model as the report writer, with LoRA.

Reads the examples built by `build_dataset` (chats of system prompt, question plus notes, and a
checked report), and trains a small adapter on top of an open model. Runs on a free Colab or
Kaggle GPU (4-bit, so an 8B model fits in 16 GB) or on a cluster GPU.

    python -m scholarcrew.finetune.train_lora --data data/sft/train.jsonl --model Qwen/Qwen3-8B

The heavy libraries (torch, transformers, peft) are imported only when training starts, so the
data handling here can be tested without them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_examples(path: Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()  # tolerates a Windows BOM
    return [json.loads(line) for line in lines if line.strip()]


def split_holdout(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Set some examples aside to watch for overfitting. Deterministic for a given seed, so a run
    can be repeated exactly and different seeds give different splits."""
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    count = round(len(rows) * fraction) if fraction > 0 else 0
    count = min(max(count, 1 if fraction > 0 and len(rows) >= 10 else 0), len(rows) - 1)
    return shuffled[count:], shuffled[:count]


def encode(row: dict, tokenizer, max_len: int) -> dict | None:
    """Turn one chat into token ids, with the loss counted on the report only.

    The prompt is built exactly as it is at inference time (the chat template with the assistant
    turn opened and Qwen3's thinking turned off), so the model is trained on what it will see.
    Returns None when the example is longer than `max_len` rather than cutting a report short.
    """
    system, user, assistant = row["messages"]
    prompt = tokenizer.apply_chat_template(
        [system, user], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(assistant["content"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + answer_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + answer_ids}


def encode_all(rows: list[dict], tokenizer, max_len: int) -> tuple[list[dict], int]:
    encoded = [encode(r, tokenizer, max_len) for r in rows]
    return [e for e in encoded if e], sum(e is None for e in encoded)


def make_collator(pad_id: int):
    """Pad a batch to its longest example. Padding is ignored by attention and by the loss."""

    def collate(batch: list[dict]) -> dict:
        import torch

        longest = max(len(b["input_ids"]) for b in batch)

        def pad(key: str, value: int) -> list[list[int]]:
            return [b[key] + [value] * (longest - len(b[key])) for b in batch]

        return {
            "input_ids": torch.tensor(pad("input_ids", pad_id)),
            "labels": torch.tensor(pad("labels", -100)),
            "attention_mask": torch.tensor(
                [[1] * len(b["input_ids"]) + [0] * (longest - len(b["input_ids"])) for b in batch]
            ),
        }

    return collate


def train(args: argparse.Namespace) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    rows = load_examples(Path(args.data))
    train_rows, val_rows = split_holdout(rows, args.val_fraction, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_set, too_long = encode_all(train_rows, tokenizer, args.max_len)
    val_set, _ = encode_all(val_rows, tokenizer, args.max_len)
    print(f"examples: {len(train_set)} train, {len(val_set)} held out, {too_long} too long and skipped")

    on_gpu = torch.cuda.is_available()
    use_bf16 = on_gpu and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if on_gpu else torch.float32)
    load_args: dict = {}
    if args.four_bit:
        load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        load_args["device_map"] = {"": 0}
    else:
        load_args["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_args)
    if args.four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    # Warm up over the first 5% of updates. Counted in steps, because the ratio setting was removed
    # from recent versions of transformers.
    updates = -(-len(train_set) // (args.batch_size * args.grad_accum)) * args.epochs
    options = {
        "output_dir": str(Path(args.out) / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": max(1, int(0.05 * updates)),
        "logging_steps": 5,
        "save_strategy": "no",
        "bf16": use_bf16,
        "fp16": on_gpu and not use_bf16,
        "optim": "paged_adamw_8bit" if args.four_bit and on_gpu else "adamw_torch",
        "report_to": "none",
        "seed": args.seed,
        "remove_unused_columns": False,
        "use_cpu": not on_gpu,
    }
    if val_set:
        options["eval_strategy"] = "epoch"
    try:
        arguments = TrainingArguments(**options)
    except TypeError:  # older transformers call it evaluation_strategy
        if "eval_strategy" in options:
            options["evaluation_strategy"] = options.pop("eval_strategy")
        arguments = TrainingArguments(**options)

    trainer = Trainer(
        model=model, args=arguments, train_dataset=train_set, eval_dataset=val_set or None,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    trainer.train()

    out = Path(args.out)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    report = {
        "base_model": args.model,
        "examples_train": len(train_set),
        "examples_held_out": len(val_set),
        "skipped_too_long": too_long,
        "settings": {k: v for k, v in vars(args).items() if k not in ("push_to_hub",)},
        "log": trainer.state.log_history,
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved the adapter and training_report.json to {out}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)
        print(f"Pushed the adapter to {args.push_to_hub} (private)")
    if val_rows and args.sample:
        show_sample(model, tokenizer, val_rows[0])
    return report


def show_sample(model, tokenizer, row: dict) -> None:
    """Write one report for a held-out question so the result can be read with the eyes."""
    import torch

    try:
        system, user, _ = row["messages"]
        prompt = tokenizer.apply_chat_template(
            [system, user], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        model.config.use_cache = True
        model.eval()
        with torch.no_grad():
            tokens = model.generate(**batch, max_new_tokens=700, do_sample=False)
        print("\n--- sample report from a held-out question ---")
        print(tokenizer.decode(tokens[0][batch["input_ids"].shape[1]:], skip_special_tokens=True))
    except Exception as exc:  # noqa: BLE001 - a failed preview must not hide a finished training run
        print(f"(could not generate a sample: {type(exc).__name__}: {exc})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scholarcrew-train", description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data/sft/train.jsonl", help="examples from build_dataset")
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="base model on Hugging Face")
    p.add_argument("--out", default="outputs/writer-lora", help="where the adapter is saved")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8, help="batches combined per update")
    p.add_argument("--max-len", type=int, default=4096, help="longer examples are skipped")
    p.add_argument("--val-fraction", type=float, default=0.1, help="share held out to watch overfitting")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-4bit", dest="four_bit", action="store_false",
                   help="load the model in full precision (needs a big GPU, or a tiny model)")
    p.add_argument("--no-sample", dest="sample", action="store_false",
                   help="skip printing a sample report after training")
    p.add_argument("--push-to-hub", metavar="USER/REPO", help="also upload the adapter, privately")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
