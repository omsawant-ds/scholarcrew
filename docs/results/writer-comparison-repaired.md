## Writer comparison

Base model: `Qwen/Qwen3-4B` | Judge: `qwen/qwen3.8-27b` | 2026-11-15

| Metric | Copy the notes | Base model | Tuned (LoRA) |
|---|---|---|---|
| Faithfulness | 81% (n=29) | 61% (n=60) | 61% (n=59) |
| Numeric grounding | 70% (n=22) | 83% (n=40) | 94% (n=44) |
| Relevance | 98% (n=30) | 92% (n=60) | 98% (n=60) |
| Citation validity | 97% (n=30) | 100% (n=60) | 100% (n=60) |
| Citation coverage | 49% (n=30) | 47% (n=60) | 88% (n=60) |
| Source utilization | 52% (n=30) | 49% (n=60) | 56% (n=60) |
| Structure | 82% (n=30) | 100% (n=60) | 100% (n=60) |

Reports scored: Copy the notes 30 (0 failed); Base model 60 (0 failed); Tuned (LoRA) 60 (0 failed).

Tuned against base, question by question (each question's score is the mean of its samples):

| Metric | Questions | Tuned better | Tied | Tuned worse | Mean difference |
|---|---|---|---|---|---|
| Faithfulness | 30 | 13 | 2 | 15 | +0.0 points |
| Relevance | 30 | 9 | 21 | 0 | +5.0 points |
| Citation coverage | 30 | 28 | 1 | 1 | +40.6 points |
