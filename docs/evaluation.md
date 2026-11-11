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
| **Citation validity** | Share of cited source numbers (`[3]`, `[1, 4]`, `[18-20]`, and full-width `【3】`) that point at a source that was actually retrieved. | No |
| **Citation coverage** | Share of substantive statements that carry a citation. | No |
| **Source utilization** | Share of retrieved sources the report cites. | No |
| **Structure** | Title, at least two sections, a references list, and enough length. | No |

A judge that cannot answer, or returns something unparseable, produces "no score" and is left out
of the mean (the table shows `n`, the number of reports each mean covers). It is never counted as
zero or as a pass.

## Results: one change kept, one tried and dropped

The first evaluation runs showed the agents **inventing statistics**. In one report, 8 of 12
statistics were not in the source they cited, and figures such as "0.71 to 0.84" appeared in no
retrieved source at all. Two causes were found: the analyst saw only the first 700 characters of
each abstract, which cuts off the results, and no prompt told the agents to avoid inventing
numbers. Two changes were then measured:

1. **Improved prompts (kept):** the analyst sees the full abstract (1200 characters), and the
   analyst and writer prompts forbid figures that are not in the sources.
2. **Writer self-check (tried and dropped):** after drafting, the writer checked its own draft with
   the same rules used for scoring. If the draft cited nothing, or stated a figure that is not in
   its cited source, it got one targeted revision and kept it only if it was better. It was
   implemented in the first commit of pull request #3 and removed in the same pull request.

Same 5 questions, same generator (`gpt-oss-20b`), same judge (`qwen3.8-27b`, a different model
family), identical scoring rules. Full tables: [old agents](results/baseline.md),
[improved prompts](results/improved.md), [prompts + self-check](results/self-check.md).

| Metric | Old agents | Improved prompts | Prompts + self-check (dropped) |
|---|---|---|---|
| Faithfulness (judged) | 43% | **71%** | 58% |
| Numeric grounding | 0% (n=1) | **89%** (n=3) | 100% (n=3) |
| Citation coverage | 34% | **45%** | 49% |
| Citation validity | 100% | 100% | 100% |
| Source utilization | 58% | 43% | 67% |
| Relevance (judged) | 100% | 100% | 100% |
| Structure | 100% | 100% | 100% |
| Mean time per report | 49s | 63s | 88s |

**Improved prompts.** Faithfulness rose on 3 of 5 questions by large margins (30% to 73%, 20% to
75%, 0% to 75%) and fell on the other two (100% to 83%, 67% to 50%). Few reports state any
statistics, so numeric grounding rests on very small counts (n is the number of reports that
contained a statistic).

**Writer self-check: not shown to help, so it was dropped.** Faithfulness was lower with it (58%
against 71%), the difference is within the run-to-run noise described below, and reports took
about 25 seconds longer. It worked as designed: it cost nothing when a draft was clean, and in one
live case it removed an unsupported figure. But it adds a model call, and on a free tier the
per-minute token limit is exactly what makes calls fail, so a feature that adds calls needs a
measured benefit and this one had none. It can be revisited if a larger test (all 30 questions,
repeated runs) shows a gain.

**What this does and does not show**

- **Noise is large.** The same code produced a faithfulness of 73% on one question in one run and
  25% in another, on a report the self-check did not touch. Every comparison here uses one generation
  per question, so differences of 10 to 20 points between columns should not be trusted. Only
  the old-agents versus improved-prompts gap, which is larger and consistent across questions, is
  reasonably convincing.
- Five questions is a small sample and no significance test was run. A fair test needs all 30
  questions and repeated runs, which free-tier daily limits make a multi-day job.
- Relevance scored 100% for every report, so this judge cannot tell the reports apart on that
  metric.
- The improved prompts cost more tokens per call and slowed reports by about 14 seconds.

### Correction

An earlier version of this page reported different numbers (faithfulness 22% before and 56%
after) and said that about 1 report in 5 came back with no citations. Both were wrong. Some
reports wrote their citations in full-width brackets, `【13】`, and the scorer did not recognize
them, so those reports were counted as uncited and their figures as unsupported. The scorer now
accepts these brackets, the writer rewrites them as ordinary `[13]`, and every number above was
re-scored from the saved reports. The bug was found by tracing a report that scored 100%
faithful while the scorer flagged 14 of its figures as unsupported.

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
- **The scorer is code, and code has bugs.** The citation-bracket bug above is an example. When a
  score looks surprising, inspect the individual claims before trusting it.
