# MultiHop-RAG — five adaptive-question conditions

609-article MultiHop-RAG corpus (`data/raw/multihoprag/`), 1,830 chunks
(1024/128), 2,255 fully aligned evidence-bearing queries. Generator
`gpt-5.4-mini`, Iris dim-384 embeddings, ASCII BM25, RRF k=60.

Verified MRR@10: 0.644 (baseline) → 0.670 (5–20) → 0.670 (unbounded) →
0.672 (chunk + article 5–20) → 0.671 (chunk + article unbounded).

## Files and run order

The condition scripts share one `results/mhrag_adaptive_questions_full/metrics.json`
and one report, so the order is strict. `run.py` enforces it:

```bash
python exp/multihoprag/run.py
```

| Step | Script | Writes |
|------|--------|--------|
| 1 | `baseline.py` | condition 1; seeds chunk/query vectors and `results/mhrag_full_baseline_1024_128/` + its own report |
| 2 | `adaptive_questions.py` | conditions 1–3 (re-injects the baseline row) |
| 3 | `chunk_article_questions.py` | appends condition 4 |
| 4 | `chunk_article_unbounded.py` | appends condition 5, renders the final five-row report |

`adaptive_lib.py` is the shared library (fact extraction, question generation,
resumable embedding, evaluation, report rendering). It is configured by
module-level assignment: `adaptive_questions.py` repoints its paths at the
full-corpus caches at import time, and `exp/yettel_bg/` and
`exp/qwen_generator/` repoint it at their own corpora the same way. Renaming
its module attributes will silently break those consumers.

## Caches

- `data/processed/mhrag_full_baseline_1024_128/` — chunks, queries, alignment
- `data/processed/mhrag_adaptive_questions_full/` — the LLM generation caches
  (`adaptive_generations.jsonl`, `article_question_generations*.jsonl`) —
  expensive; do not delete
- `results/mhrag_adaptive_questions_full/*_iris.json` — vector caches
  (gitignored); with them present a full re-run is offline
