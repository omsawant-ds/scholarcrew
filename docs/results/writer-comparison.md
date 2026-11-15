## Writer comparison

Base model: `Qwen/Qwen3-4B` | Judge: `qwen/qwen3.8-27b` | 2026-11-14

| Metric | Copy the notes | Base model | Tuned (LoRA) |
|---|---|---|---|
| Faithfulness | 84% (n=30) | 55% (n=60) | 48% (n=60) |
| Numeric grounding | 71% (n=22) | 75% (n=40) | 76% (n=44) |
| Relevance | 98% (n=30) | 92% (n=60) | 95% (n=60) |
| Citation validity | 97% (n=30) | 100% (n=60) | 100% (n=60) |
| Citation coverage | 49% (n=30) | 47% (n=60) | 88% (n=60) |
| Source utilization | 52% (n=30) | 43% (n=60) | 52% (n=60) |
| Structure | 82% (n=30) | 100% (n=60) | 100% (n=60) |

Reports scored: Copy the notes 30 (0 failed); Base model 60 (0 failed); Tuned (LoRA) 60 (0 failed).

Tuned against base, question by question (each question's score is the mean of its samples):

| Metric | Questions | Tuned better | Tied | Tuned worse | Mean difference |
|---|---|---|---|---|---|
| Faithfulness | 30 | 8 | 2 | 20 | -7.2 points |
| Relevance | 30 | 6 | 20 | 4 | +2.1 points |
| Citation coverage | 30 | 28 | 1 | 1 | +40.6 points |
