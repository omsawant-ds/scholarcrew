# Evaluation

ScholarCrew ships with an automated evaluation pipeline so changes to prompts, models or agents
can be measured instead of guessed at.

```bash
python -m scholarcrew.evaluation --quick                  # 5 questions
python -m scholarcrew.evaluation --n 10                   # first 10
python -m scholarcrew.evaluation --topic RAG              # one topic
python -m scholarcrew.evaluation --model <other-model>    # compare generators
python -m scholarcrew.evaluation --rescore eval_results/<run>.json --judge-model <other-model>
python -m scholarcrew.evaluation --resume eval_results/<run>.json   # re-run only failed questions
python -m scholarcrew.evaluation --min-faithfulness 0.7   # exit code 1 if the mean is lower
```

Each run saves a JSON file (every report, its sources and scores) and a Markdown summary in
`eval_results/`. `--rescore` re-judges saved reports without regenerating them, so changing the
judge or a metric costs a fraction of a full run.

## Question set

30 research questions in 12 topics (RAG, agents, fine-tuning, evaluation, safety, inference,
retrieval, prompting, architecture, multimodal, data, training), in
[`questions.json`](../src/scholarcrew/evaluation/questions.json).

## Metrics

All scores are between 0 and 100%. Higher is better.

| Metric | How it is computed | Needs a model? |
|---|---|---|
| **Faithfulness** | Up to 12 cited statements are sampled from the report. A judge model checks each one against the text of the sources it cites. Score = share supported. A report that cites nothing scores 0%. | Yes |
| **Relevance** | A judge rates 1 to 5 how directly and completely the report answers the question, rescaled to 0 to 100%. | Yes |
| **Numeric grounding** | Every statistic (a percentage or decimal) in the report must appear in the source it cites. Uncited statistics count as ungrounded. A cheap check for invented figures. | No |
| **Citation validity** | Share of cited source numbers (`[3]`, `[1, 4]`, `[18-20]`) that point at a source that was actually retrieved. | No |
| **Citation coverage** | Share of substantive statements that carry a citation. | No |
| **Source utilization** | Share of retrieved sources the report cites. | No |
| **Structure** | Title, at least two sections, a references list, and enough length. | No |

A judge that cannot answer, or returns something unparseable, produces "no score" and is left out
of the mean (the table shows `n`, the number of reports each mean covers). It is never counted as
zero or as a pass.

## Results: measuring a fix

The first evaluation runs showed the agents **inventing statistics**. In one report, 8 of 12
statistics were not in the source they cited, and figures such as "0.71 to 0.84" appeared in no
retrieved source at all. Two causes were found: the analyst saw only the first 700 characters of
each abstract, which cuts off the results, and no prompt told the agents to avoid inventing
numbers. The fix gives the analyst the full abstract (1200 characters) and adds explicit rules
about figures to the analyst and writer prompts.

Same 5 questions, same generator (`gpt-oss-20b`), same judge (`qwen3.8-27b`, a different model
family), scored with identical rules. Full tables:
[baseline](results/baseline.md) and [improved](results/improved.md).

| Metric | Before | After |
|---|---|---|
| Faithfulness (judged) | 22% | **56%** |
| Numeric grounding | 0% (n=1) | **56%** (n=3) |
| Citation coverage | 21% | **32%** |
| Citation validity | 80% | 80% |
| Source utilization | 42% | 34% |
| Relevance (judged) | 100% | 100% |
| Structure | 100% | 100% |
| Mean time per report | 49s | 63s |

Per question, faithfulness improved on 3 of 5 (0% to 83%, 30% to 73%, 0% to 75%), dipped on one
(58% to 50%), and fell on one (20% to 0%) where the writer produced a report with no citations.

**What this does and does not show**

- Five questions is a small sample. The direction is encouraging, but a gap this size could
  partly be noise, and no significance test was run.
- About 44% of statistics are still not found in their cited source, so the fix reduced invented
  figures without eliminating them.
- In both versions, 1 report in 5 came back with no citations at all, so the writer sometimes
  ignores the citation instruction. This is the next thing to fix.
- Relevance scored 100% for every report in both versions, so this judge cannot tell the reports
  apart on that metric.
- The fix costs more tokens per call and slowed reports by about 14 seconds.

## Running on free tiers

Free API tiers cap tokens per minute and per day, and this shaped the design:

- A question where any model call fails is recorded as **failed** and left out of the averages.
  Earlier, reports written after a rate-limit error were scored as if the system had worked.
- A run stops after 3 failures in a row instead of burning the remaining quota.
- The judge waits and retries when it gets an empty reply, and the app's client waits out
  provider rate limits.
- `--resume` finishes a partial run, and refuses to mix different models in one result.
- A run of about 5 questions costs on the order of 100,000 tokens, so a 200,000-token daily limit
  fits roughly one or two runs per model per day. Use different models for generating and judging
  to spread the load across separate quotas.

## Limitations

- **Judge bias.** By default the judge is the same model as the generator, which tends to rate
  its own family generously. Use `--judge-model` to judge with a different model, as the results
  above do.
- **Sources are abstracts and snippets.** The agents read arXiv abstracts and search snippets, not
  full papers. Faithfulness measures support in that retrieved text, not in the whole paper.
- **Numeric grounding is a proxy.** It checks that a number appears in the cited text. It can flag
  a legitimate unit conversion (0.84 vs 84%) and cannot tell whether a matching number is used
  correctly.
- **Small samples.** Free API tiers limit how many questions a run can cover. Treat differences of
  a few points between runs as noise.
