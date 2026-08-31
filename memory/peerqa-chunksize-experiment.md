---
name: peerqa-chunksize-experiment
description: PeerQA chunk-size × question-enrichment study; bigger chunks retrieve better and want more questions, but plain large chunk-text beats questions-only; adaptive length-based beats fixed q10.
metadata:
  type: project
---

Follow-up to [[peerqa-questions-only-experiment]]. Question: does the amount of
generated-question enrichment needed depend on chunk size / information density?
Code: `src/peerqa/chunksize_experiment.py`, `run_chunksize.py`, `chunksize_html.py`
(reuses peerqa_data + peerqa_experiment; generation cached to
`data/processed/peerqa/chunksize_questions.jsonl`; results
`results/peerqa/chunksize_results.json`; report
`report/peerqa_chunksize_results.html`). Same 15-paper subset, 65 queries,
Octen-0.6B, gpt-4o-mini. Generation up to 15 grounded questions/chunk once, sliced.

Design: **Table 1** = 4 fixed chunkings (200/400/600/800 tok, ~22% overlap) ×
q5/q10/q13/q15 questions-only + per-size chunk-text baseline. **Table 2** = one
section-aware VARIABLE chunking (597 chunks, 1–801 tok) comparing fixed q10 vs
adaptive length-based (#q from chunk token length, bands 200→4…800→12) vs adaptive
density-based (regex signals: numbers/acronyms/cap-phrases/struct-refs/metric-terms,
percentile→[4,15], calibrated avg ~9.5 = matched budget) vs fused (chunk vector ⊕
best-question score). **Table 3** = best/cheapest-strong/fastest/best-enrichment.

**Key results (nDCG@10):**
- **Chunk size is the dominant lever, and bigger = better here.** Chunk-text
  baselines rise monotonically: 200t 0.482 → 400t 0.522 → 600t 0.571 → **800t 0.605**.
  Size moves nDCG ~0.12; question count within a size moves it up to ~0.16 but
  mostly BELOW baseline.
- **Bigger chunks want more questions** (best fixed count: 200t→q10, 400t→q15,
  600t→q13, 800t→q15). Small chunks saturate early (q5≈q15 at 200t; LLM can't make
  15 distinct grounded Qs from ~200 tok, extras become near-dup embeddings).
- **Questions-only never beats chunk-text on nDCG@10** at any size — consistent
  with [[peerqa-questions-only-experiment]] (questions-only wins recall Hit@5/10,
  loses rank-1 precision).
- **Adaptive length-based beats fixed q10**: on variable chunking 0.537 vs q10 0.501,
  at FEWER embeddings (3725 vs 5795) — reallocating budget by length wins cheaply.
  Density-based 0.509 ≈ q10. **Fused = 0.551 (best enrichment), recovers precision.**
- Overall best/cheapest/fastest all = **800t plain chunk-text baseline** (0.6048, 8.4MB).

**Recommendation given:** use large (800t) chunks plain; if enriching, keep the
chunk vector and FUSE (don't go questions-only), prefer length-based adaptive over
density; questions-only only for recall-at-depth. Revisit at full-corpus scale.
Cost: ~2555 chunks generation, a few min, <$1.
