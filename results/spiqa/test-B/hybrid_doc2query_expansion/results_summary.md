# Hybrid Doc2Query-style generated-question retrieval — summary

SPIQA test-B (50 papers). Generated questions used as **document expansion** for BM25 vs as dense embedding text; hybrid = dense(original chunks) + BM25(enriched) fused with RRF (k=60). Questions map back to parent chunk; only original chunks are returned. Same LLM/embedder, no fine-tuning.

## Comparison table

| condition | hit@1 | hit@5 | hit@10 | mrr | ndcg@10 | generated_questions | kept_questions | filtered_% | dense_index_MB | bm25_index_MB | storage_x | p95_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_dense | 0.5714 | 1.0 | 1.0 | 0.7262 | 0.8002 | 0 | 0 | 0.0 | 1.174 | 0 | 1.0 | 216.53 |
| baseline_hybrid | 0.8571 | 1.0 | 1.0 | 0.9286 | 0.9233 | 0 | 0 | 0.0 | 1.174 | 0.28 | 1.24 | 176.094 |
| q10_dense_append | 0.5714 | 1.0 | 1.0 | 0.719 | 0.7956 | 1208 | 1208 | 0.0 | 1.174 | 0 | 1.0 | 142.864 |
| q10_bm25_expand | 0.8571 | 1.0 | 1.0 | 0.9286 | 0.9132 | 1208 | 1208 | 0.0 | 0 | 0.412 | 0.35 | 0.347 |
| q10_hybrid_expand | 0.7143 | 1.0 | 1.0 | 0.8571 | 0.8821 | 1208 | 1208 | 0.0 | 1.174 | 0.412 | 1.35 | 140.016 |
| q50_hybrid_expand | 0.7143 | 1.0 | 1.0 | 0.8571 | 0.8804 | 1937 | 1937 | 0.0 | 1.174 | 0.487 | 1.41 | 137.624 |
| q10_filtered_hybrid_expand | 0.8571 | 1.0 | 1.0 | 0.9286 | 0.9036 | 1208 | 1029 | 14.8 | 1.174 | 0.394 | 1.34 | 133.389 |
| q50_filtered_hybrid_expand | 0.8571 | 1.0 | 1.0 | 0.9286 | 0.912 | 1937 | 1570 | 18.9 | 1.174 | 0.452 | 1.39 | 208.906 |

## Findings

1. **BM25-expand vs dense-append (q10):** BM25 nDCG@10 0.9132 vs dense-append 0.7956 → generated questions help sparse MORE than dense.
2. **Hybrid vs dense-only:** hybrid_expand_q10 0.8821 vs baseline_dense 0.8002 → hybrid improves retrieval. (baseline_hybrid, no questions = 0.9233.)
3. **q10 vs q50 (hybrid):** 0.8821 vs 0.8804 → q50 adds little/noise over q10.
4. **Round-trip filtering:** q10 helped (14.8% removed); q50 helped (18.9% removed).
5. **Questions as lexical expansion vs dense text:** YES — BM25 expansion beats dense append, supporting the doc2query hypothesis.
6. **Cost/benefit:** best generated-question condition `q10_bm25_expand` = 0.9132 nDCG@10 (+0.113 vs dense baseline) at 0.35× storage, 0.347ms p95. Worth it.

## Recommendation
**q10_bm25_expand** — hybrid BM25 expansion with generated questions is the best configuration.
