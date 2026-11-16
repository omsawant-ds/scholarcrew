# Evaluation

```bash
python -m scholarcrew.evaluation --quick                  # 5 questions
python -m scholarcrew.evaluation --n 10                   # first 10
python -m scholarcrew.evaluation --topic RAG              # one topic
python -m scholarcrew.evaluation --model <model>          # generator
python -m scholarcrew.evaluation --judge-model <model>    # judge (default: the generator)
python -m scholarcrew.evaluation --rescore <run>.json     # re-judge saved reports
python -m scholarcrew.evaluation --resume <run>.json      # re-run failed questions only
python -m scholarcrew.evaluation --min-faithfulness 0.7   # exit 1 if the mean is lower
```

Each run saves every report, its sources and scores to `eval_results/` (JSON and Markdown).
The question set is 30 questions in 12 AI topics:
[`questions.json`](../src/scholarcrew/evaluation/questions.json).

## Metrics

Scores are 0 to 100%; higher is better.

| Metric | Definition | Judge |
|---|---|---|
| Faithfulness | Up to 12 cited statements per report are checked against the excerpt of the source they cite. Share supported. A report with no citations scores 0%. | Yes |
| Relevance | 1 to 5 rating of how well the report answers the question, rescaled. | Yes |
| Numeric grounding | Statistics in the report that appear in the source they cite. | No |
| Citation validity | Citation numbers that point at a retrieved source. | No |
| Citation coverage | Substantive statements that carry a citation. | No |
| Source utilization | Retrieved sources the report cites. | No |
| Structure | Title, at least two sections, references, minimum length. | No |

A judge failure is "no score": it is excluded from the mean and shown in `n`, never counted as 0
or as a pass. A question where any model call fails is recorded as failed and excluded. A run
stops after 3 consecutive failures.

## Prompt changes

Same 5 questions, generator `gpt-oss-20b`, judge `qwen3.8-27b`. Tables:
[old agents](results/baseline.md), [improved prompts](results/improved.md),
[prompts + self-check](results/self-check.md).

| Metric | Old agents | Improved prompts | Prompts + self-check (dropped) |
|---|---|---|---|
| Faithfulness | 43% | **71%** | 58% |
| Numeric grounding | 0% (n=1) | **89%** (n=3) | 100% (n=3) |
| Citation coverage | 34% | **45%** | 49% |
| Mean time per report | 49s | 63s | 88s |

- **Kept:** the analyst sees the full abstract (1200 characters, was 700) and the prompts forbid
  figures not in the sources.
- **Dropped:** a writer self-check showed no gain (faithfulness 58% against 71%), added about 25
  seconds and a model call.
- Five questions, one run each. Run-to-run noise is large (one question scored 73% and 25% on
  identical code); only the old versus improved gap is consistent across questions.

## Fine-tuned writer

**Setup.** LoRA (r=16) on Qwen3-4B, 4-bit, one run on a free Kaggle T4: 86 training examples
written by `gpt-oss-120b` and filtered by rule-based checks, 10 held out, 3 epochs, 107 minutes.
Held-out loss 0.658, 0.584, 0.572.

**Comparison.** 30 held-out questions, identical frozen sources and notes for every writer, 2
samples per model, judge `qwen3.8-27b`. Writers: the base model, the tuned model, and a baseline
that returns the analyst's notes unchanged. The 118 training questions are disjoint from the 30.

| Metric | Copy the notes | Qwen3-4B | Qwen3-4B + LoRA |
|---|---|---|---|
| Faithfulness, as written | 84% | 55% | 48% |
| Faithfulness, citations repaired | 81% | 61% | 61% |
| Citation coverage | 49% | 47% | 88% |
| Source utilization (repaired) | 52% | 49% | 56% |
| Relevance (repaired) | 98% | 92% | 98% |
| Numeric grounding (repaired) | 70% (n=22) | 83% (n=40) | 94% (n=44) |

Tuned against base, faithfulness by question: as written 8 better, 2 tied, 20 worse (sign test
p of about 0.04); repaired 13 better, 2 tied, 15 worse. Full tables:
[as written](results/writer-comparison.md), [repaired](results/writer-comparison-repaired.md).

**Findings**

- Tuning raised citation coverage (47% to 88%) and not faithfulness.
- On 242 judged claims from 22 reports, claims citing only source [1] were supported 31% to 32%
  of the time, against 62% to 74% for all others. Claims describing what the sources do not say
  were supported 29% to 40%.
- `repair_citations` points each cited statement at the source whose title and excerpt share the
  most words with it. It adds no model call, leaves statements that match no source unchanged,
  and does not alter wording, headings, tables or references. Faithfulness rose from 55% to 61%
  (base) and 48% to 61% (tuned). On 12 questions a second judge (`gpt-oss-120b`) showed the same
  direction: 60% to 71% and 49% to 64% of claims supported.
- Copying the notes remains the most faithful writer. The repair is used in the app; the tuned
  model is not.

**Limitations**

- 30 questions and 2 samples; no confidence intervals. The repaired tuned and base writers tie.
- One judge for the full run, from the same model family as the student. The second judge covers
  12 questions. The judge was not validated against human labels.
- The repair was designed after seeing these results, on these questions, and matches against the
  same 1200-character excerpt the judge reads. It needs a fresh held-out set.
- The repair matches by shared words, so it favours long sources and cannot fix a wrong citation
  when generic words dominate. The judge checks a statement against all sources it cites, so it
  cannot detect a citation attached to the wrong clause.
- The numbers above predate two later changes and have not been re-measured: statements that join
  claims with different sources are no longer repaired, and the app's writer now ends each
  statement with its own citations.
- Faithfulness is support in retrieved abstracts and snippets, not factual accuracy.
- Training examples were filtered with checks shared with the metrics, so the coverage gain is
  expected. The teacher prompt required a citation on every sentence. One training run, one
  setting.

**Reproduce.** Generation needs a GPU ([`compare_writers_kaggle.ipynb`](../notebooks/compare_writers_kaggle.ipynb));
scoring runs anywhere and resumes after a quota stop.

```bash
python -m scholarcrew.finetune.compare generate --adapter outputs/writer-lora
python -m scholarcrew.finetune.compare score --generations generations.jsonl --judge-model qwen/qwen3.8-27b
python -m scholarcrew.finetune.compare score --repair --generations generations.jsonl --judge-model qwen/qwen3.8-27b --out-dir data/compare-repaired
```

## Strict mode

Optional: a checkbox in the app, `--strict` on the command line. It keeps only cited statements for
which a sentence of the cited source contains at least 40% of the statement's key words, and whose
numbers all appear in the source. It makes no model call.

The app's writer cites only statements of fact taken from the notes; its own comparisons and
conclusions are left uncited. Against a writer that cited every statement, the judge supported 75%
of cited statements instead of 67% (73% against 66% on the 18 questions not used for tuning).

Statements in a Limitations section are never filtered: they are the writer's remarks about the
evidence, and a source does not state them. They were 15% of the cited statements, and strict mode
used to delete 71% of them.

Measured on the 30 evaluation questions (326 cited statements outside Limitations sections), with
the app's writer (`gpt-oss-120b`) and judge `qwen3.8-27b`. The threshold was chosen on 12 of the
questions; the other 18 give the same result (72% kept, 90% supported at 0.4).

| Threshold | Statements kept | Supported by the judge, among those kept |
|---|---|---|
| none | 100% | 80% |
| 0.3 | 88% | 86% |
| **0.4 (used; table rows 0.3)** | **76%** | **90%** |
| 0.5 | 60% | 93% |
| 0.6 | 40% | 94% |

- Uncited sentences are the writer's own text. Neither the judge nor strict mode checks them.
- Table rows are checked as whole rows at 0.3, because they are short and already well supported
  (92% judge-supported). That rests on 37 rows, a small sample.
- A text match is not proof: about 40% of the statements dropped at 0.4 were supported by the judge.
- A small model (`gpt-oss-20b`) judging each claim against its best-matching sentence kept 42% at
  90% supported, no better than the word match at the same share. Requiring both kept 35% at 95%.
- Supported means supported by the judge, not true.

## Limitations

- The judge is an LLM; use `--judge-model` with a different family from the generator.
- Sources are abstracts and snippets, not full papers.
- Numeric grounding is a proxy: it checks that a number appears in the cited text, not that it is
  used correctly.
- Free tiers cap daily tokens, which limits sample size.
