# Combined corpus — union-interference study

Merges both corpora into one 949-document / 2,793-chunk index and evaluates
conditions 1–4 with all 4,510 evidence-bearing queries, so each dataset acts as
a distractor corpus for the other. IDs are namespaced (`mhrag::`, `yettel::`);
all cached generations and vectors are reused byte-for-byte.

**These are not the headline per-dataset numbers.** With Yettel distractors
present, the MultiHop-RAG baseline MRR@10 is 0.641 here vs 0.644 standalone —
the per-dataset tables in the presentation come from `exp/multihoprag/` and
`exp/yettel_bg/`. Cross-corpus leakage at rank 1 is near zero, so read this as
a robustness check, not a harder benchmark.

```bash
python exp/combined/run_combined_experiments.py   # evaluate + basic report
python exp/combined/combined_report.py            # rich report: deltas, interference, leakage
```

Outputs: `results/combined_mhrag_yettel_experiments/{metrics.json,rankings.jsonl}`,
`report/combined_multihop_yettel_adaptive_questions.html`, and
`report/combined_multihop_yettel_four_experiments.html`.
