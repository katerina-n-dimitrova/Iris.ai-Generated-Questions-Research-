# Vision-enriched generated questions + BM25 / hybrid — SPIQA test-B (20 papers)

Retrieval on **20 papers · 1026 chunks · 74 eval queries** (74 figure-answerable), three modes per condition: **dense**, **BM25**, and **hybrid** (dense(original) + BM25 fused with RRF, k=60). Generated questions are document expansion (chunk + first-n questions). **216** figure/table units, **216** vision-enriched; the **vision** arm folds the gpt-4o-mini visual description into the figure chunk before it is embedded, BM25-indexed, and used to generate questions. Embedder huggingface:Octen/Octen-Embedding-0.6B:dim1024; no fine-tuning; questions + vision cached (no new LLM calls).

*(Every SPIQA test-B question refers to a figure/table, so all 74 queries are figure-answerable — the vision enrichment can affect any of them.)*

## novision arm

| condition | hit@1 | hit@5 | hit@10 | mrr | ndcg@10 | generated_questions | dense_index_MB | bm25_index_MB | storage_x | p95_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_dense | 0.4459 | 0.7568 | 0.8108 | 0.5696 | 0.6128 | 0 | 9.328 | 0.0 | 1.0 | 49.07 |
| baseline_bm25 | 0.5541 | 0.7703 | 0.8378 | 0.6432 | 0.664 | 0 | 0.0 | 2.23 | 0.24 | 2.099 |
| baseline_hybrid | 0.473 | 0.8108 | 0.8378 | 0.607 | 0.6616 | 0 | 9.328 | 2.23 | 1.24 | 33.734 |
| q10_dense_append | 0.5 | 0.7838 | 0.8378 | 0.6285 | 0.6587 | 10191 | 9.332 | 0.0 | 1.0 | 30.642 |
| q10_bm25_expand | 0.5405 | 0.8243 | 0.8784 | 0.6634 | 0.6933 | 10191 | 0.0 | 3.334 | 0.36 | 2.046 |
| q10_hybrid_expand | 0.5135 | 0.8108 | 0.8649 | 0.6479 | 0.6919 | 10191 | 9.328 | 3.334 | 1.36 | 33.803 |
| q50_bm25_expand | 0.5541 | 0.8514 | 0.8649 | 0.6666 | 0.6928 | 16313 | 0.0 | 3.981 | 0.43 | 2.034 |
| q50_hybrid_expand | 0.5676 | 0.7973 | 0.8514 | 0.672 | 0.6972 | 16313 | 9.328 | 3.981 | 1.43 | 33.654 |

## vision arm

| condition | hit@1 | hit@5 | hit@10 | mrr | ndcg@10 | generated_questions | dense_index_MB | bm25_index_MB | storage_x | p95_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_dense | 0.5405 | 0.7568 | 0.7973 | 0.624 | 0.6416 | 0 | 9.328 | 0.0 | 1.0 | 36.294 |
| baseline_bm25 | 0.527 | 0.7432 | 0.8378 | 0.6257 | 0.6519 | 0 | 0.0 | 2.411 | 0.26 | 2.021 |
| baseline_hybrid | 0.4865 | 0.8378 | 0.8649 | 0.6331 | 0.6764 | 0 | 9.328 | 2.411 | 1.26 | 33.933 |
| q10_dense_append | 0.5135 | 0.8108 | 0.8378 | 0.6453 | 0.6684 | 10209 | 9.328 | 0.0 | 1.0 | 31.153 |
| q10_bm25_expand | 0.5676 | 0.8378 | 0.8919 | 0.6804 | 0.7071 | 10209 | 0.0 | 3.506 | 0.38 | 2.11 |
| q10_hybrid_expand | 0.5676 | 0.8108 | 0.8784 | 0.6759 | 0.7125 | 10209 | 9.328 | 3.506 | 1.38 | 33.662 |
| q50_bm25_expand | 0.527 | 0.8649 | 0.8784 | 0.6671 | 0.7077 | 17799 | 0.0 | 4.299 | 0.46 | 2.182 |
| q50_hybrid_expand | 0.5946 | 0.8378 | 0.9054 | 0.6954 | 0.7258 | 17799 | 9.328 | 4.299 | 1.46 | 33.894 |

## Vision effect (vision − novision nDCG@10), per condition

| condition | novision nDCG@10 | vision nDCG@10 | Δ nDCG@10 | Δ Hit@5 |
| --- | --- | --- | --- | --- |
| baseline_dense | 0.6128 | 0.6416 | +0.0288 | +0.0 |
| baseline_bm25 | 0.664 | 0.6519 | -0.0121 | -0.0271 |
| baseline_hybrid | 0.6616 | 0.6764 | +0.0148 | +0.027 |
| q10_dense_append | 0.6587 | 0.6684 | +0.0097 | +0.027 |
| q10_bm25_expand | 0.6933 | 0.7071 | +0.0138 | +0.0135 |
| q10_hybrid_expand | 0.6919 | 0.7125 | +0.0206 | +0.0 |
| q50_bm25_expand | 0.6928 | 0.7077 | +0.0149 | +0.0135 |
| q50_hybrid_expand | 0.6972 | 0.7258 | +0.0286 | +0.0405 |

## Findings

- **Does BM25 help? (doc2query)** dense baseline nDCG@10 0.6128 vs BM25 baseline 0.664 vs hybrid baseline 0.6616 (novision). With q10 expansion: BM25 0.6933, hybrid 0.6919 — hybrid beats the dense baseline.
- **Best novision config:** `q50_hybrid_expand` = 0.6972 nDCG@10. **Best vision config:** `q50_hybrid_expand` = 0.7258 nDCG@10.
- **Vision effect on hybrid retrieval:** `q10_hybrid_expand` +0.0206 nDCG@10 / +0.0 Hit@5; `q50_hybrid_expand` +0.0286 nDCG@10. On plain dense chunks (`baseline_dense`) vision is +0.0288 nDCG@10; on BM25 (`q10_bm25_expand`) +0.0138.
- **Where vision lands hardest:** the visual description adds lexical anchors (axis labels, chart type, entities) that BM25 can match verbatim, and it lets the LLM write ~20 questions/figure vs ~13 caption-only — so the effect shows up in both the dense and BM25 sides of the hybrid.

## Recommendation
Use **q50_hybrid_expand** (vision-enriched hybrid). Hybrid dense+BM25 with vision-enriched doc2query questions is the strongest configuration here — BM25 contributes the lexical match the dense embedder misses, and the vision description feeds both sides.
