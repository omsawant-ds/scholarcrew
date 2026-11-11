## Evaluation results

Generator: `openai/gpt-oss-20b` | Judge: `qwen/qwen3.8-27b` | Questions: 5 (0 failed) | 2026-11-11

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 22% (n=5) | cited claims the sources support (judged) |
| Numeric grounding | 0% (n=1) | statistics that appear in the source they cite |
| Relevance | 100% (n=5) | how well the report answers the question (judged) |
| Citation validity | 80% (n=5) | citation markers that point at a real source |
| Citation coverage | 21% (n=5) | substantive statements that carry a citation |
| Source utilization | 42% (n=5) | retrieved sources the report cites |
| Structure | 100% (n=5) | title, sections, references, length |

Mean time per report: 49s. Mean sources retrieved: 19.4.

<details><summary>Per-question scores</summary>

| ID | Topic | Faithful | Relevant | Cite valid | Cite coverage | Time |
|---|---|---|---|---|---|---|
| rag-01 | RAG | 0% | 100% | 0% | 0% | 15s |
| rag-02 | RAG | 30% | 100% | 100% | 38% | 57s |
| rag-03 | RAG | 20% | 100% | 100% | 19% | 49s |
| rag-04 | RAG | 0% | 100% | 100% | 7% | 59s |
| agents-01 | Agents | 58% | 100% | 100% | 43% | 62s |

</details>
