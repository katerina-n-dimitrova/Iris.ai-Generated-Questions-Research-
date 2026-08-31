---
name: spiqa-followup-question-quality
description: Follow-up SPIQA study on generated-question QUALITY (not count); structured vision wins, most tweaks don't beat q10 hybrid.
metadata:
  type: project
---

Follow-up to [[spiqa-num-questions-experiment]]: since increasing question *count* saturates (q10 sweet spot), this study holds count ≈q10 and varies question QUALITY / indexing. Runs on the same 20-paper vision set (1026 chunks, 74 figure-answerable queries), same embedder (Octen-0.6B), same LLM (gpt-4o-mini), all on novision base except the structured arm.

**Code (src/spiqa/):** `followup_lib.py` (term extraction, heuristic+round-trip filtering, QuestionIndex, generic RRF `eval_fusion`, `eval_rerank`, CrossEncoderReranker), `followup_gen.py` (cached bm25-aware + structured-figure question generation), `followup_quality_experiment.py` (orchestrator, reuses `hybrid_doc2query_experiment` primitives), `followup_html.py` (idempotent append between `<!--FOLLOWUP_START/END-->` markers). Outputs: `results/spiqa/test-B/followup_quality/`. New caches: `data/processed/spiqa/test-Bfu20_questions_bm25aware.jsonl`, `test-Bfu20_structured_figs.jsonl` (+ reused `test-B_questions_diverse.jsonl`).

**Key results (nDCG@10; generic q10 hybrid control = 0.6919, reproduced exactly):**
- **Best new = `structured_vision_q10_hybrid` = 0.7134** (+0.1006 vs baseline_dense; beats all novision configs and prior vision q10 hybrid 0.7125). Structuring figure/table into fields (axes/values/methods/dataset) is the only clear win.
- Diverse q10 (0.6865), BM25-aware (0.6853 hybrid / 0.6761 BM25) — did NOT beat generic; generic questions already carry salient terms.
- Filtered q10 (0.6924) — matched generic with ~40% fewer questions → cost win, not quality gain.
- Separate question index RRF (0.6837) — worse than concatenation AND ~5.3× storage.
- Cross-encoder rerank (ms-marco-MiniLM) HURT nDCG (0.606) but gave highest Hit@10 (0.9054) and ~0.4–2.8s p95 — recall present, general-domain reranker mis-orders scientific text; recall (not ranking) was the bottleneck. Consistent with [[spiqa-statement-q10-findings]] (rerank hurt there too).

**Recommendation given in report:** adopt structured visual context for figure/table chunks; filter candidates for cost; skip separate-index and a general-domain reranker on this data.
