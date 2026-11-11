## Evaluation results

Generator: `openai/gpt-oss-20b` | Judge: `qwen/qwen3.8-27b` | Questions: 5 (0 failed) | 2026-11-11

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 56% (n=5) | cited claims the sources support (judged) |
| Numeric grounding | 56% (n=3) | statistics that appear in the source they cite |
| Relevance | 100% (n=5) | how well the report answers the question (judged) |
| Citation validity | 80% (n=5) | citation markers that point at a real source |
| Citation coverage | 32% (n=5) | substantive statements that carry a citation |
| Source utilization | 34% (n=5) | retrieved sources the report cites |
| Structure | 100% (n=5) | title, sections, references, length |

Mean time per report: 63s. Mean sources retrieved: 18.0.

<details><summary>Per-question scores</summary>

| ID | Topic | Faithful | Relevant | Cite valid | Cite coverage | Time |
|---|---|---|---|---|---|---|
| rag-01 | RAG | 83% | 100% | 100% | 40% | 81s |
| rag-02 | RAG | 73% | 100% | 100% | 42% | 68s |
| rag-03 | RAG | 0% | 100% | 0% | 0% | 36s |
| rag-04 | RAG | 75% | 100% | 100% | 46% | 68s |
| agents-01 | Agents | 50% | 100% | 100% | 32% | 63s |

</details>
