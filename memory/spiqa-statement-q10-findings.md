---
name: spiqa-statement-q10-findings
description: Results of the statement-level q10 enrichment experiment on SPIQA test-B (score fusion wins; filtering & rerank hurt).
metadata:
  type: project
---

Statement-level q10 experiment (src/spiqa/statement_q10_experiment.py + statement_gen.py),
full SPIQA test-B (65 papers, 3326 chunks, 228 queries, Octen-0.6B embedder, gpt-4o-mini).
Generated 16,613 statement-level anchored questions ($1.52, ~34 min at 16 workers).

Key results (nDCG@10 / Hit@1 vs baseline 0.564 / 0.408):
- **q10_statement_score_fusion = BEST: 0.622 / 0.513** (+0.058 nDCG, +0.10 Hit@1). First method to clearly beat baseline at scale.
- raw_append 0.591/0.456; filtered_score_fusion 0.599/0.474; roundtrip_filter_append 0.589; rerank 0.568 (≈baseline).

Findings (for the writeup):
- **Score fusion is the key ingredient** — search questions separately, collapse to parent by BEST question score, blend 0.7*orig+0.3*best_q. Far better than folding questions into one enriched vector (append).
- **Round-trip+anchor filtering HURT** (removed 44% incl. useful questions); the max-based fusion already suppresses noise, so filtering just costs recall.
- **Rerank-by-original-chunk HURT** — discards the fusion signal, reverts toward baseline.
- **Best question TYPES:** definition, result, numeric_detail, method. figure/table questions round-trip poorly (keep-rate 0.30/0.35).
- **Anchors help** (round-trip 62.7% anchored vs 57.7% non-anchored). Near-duplicate questions only 0.69%.
- Main residual error: correct-paper-wrong-chunk 23% (chunk granularity); when wrong, the base chunk embedding (not the questions) is at fault.
- Cost: score_fusion 4.25x storage (100MB vs 23MB), p95 1.24ms (negligible). filtered_score_fusion 2.72x for +0.035 — cheaper middle ground.

Bottom line: the encoder still caps absolute numbers, but score-fusion of anchored statement-questions is a real, cheap-latency win. See [[spiqa-num-questions-experiment]], [[spiqa-dataset-schema]].
