# q10 generated-question enrichment — results summary

SPIQA **test-B** · 0 baseline control · questions are always folded back into the parent chunk (append and/or 0.7·orig + 0.3·enriched score fusion). No fine-tuning.

## Comparison table

| condition | hit@1 | hit@5 | hit@10 | mrr | ndcg@10 | generated_questions | kept_questions | filtered_pct | embeddings | index_mb | storage_x | search_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.4459 | 0.7568 | 0.8108 | 0.5696 | 0.6128 | 0 | 0 | 0.0 | 1026 | 9.328 | 1.0 | 0.078 |
| q10_raw_append | 0.4865 | 0.7838 | 0.8378 | 0.6119 | 0.6465 | 10191 | 10191 | 0.0 | 1026 | 9.328 | 1.0 | 0.074 |
| q10_roundtrip_filter_append | 0.4324 | 0.7838 | 0.8243 | 0.5824 | 0.6315 | 10191 | 7135 | 30.0 | 1026 | 9.328 | 1.0 | 0.1 |
| q10_score_fusion | 0.4595 | 0.7703 | 0.8514 | 0.5892 | 0.631 | 10191 | 10191 | 0.0 | 2052 | 18.09 | 1.94 | 0.147 |
| q10_roundtrip_filter_score_fusion | 0.473 | 0.7703 | 0.8243 | 0.5925 | 0.6274 | 10191 | 7135 | 30.0 | 2052 | 18.09 | 1.94 | 0.131 |

## Findings

- **Best q10 method:** `q10_raw_append` (0.6465 nDCG@10, +0.0337 nDCG@10 vs baseline).
- **Round-trip filtering:** did not help (-0.015 nDCG@10, append with-vs-without filter; filtered 30.0% of questions).
- **Score fusion (0.7·orig+0.3·enriched):** did not help (-0.0155 nDCG@10 vs plain append).
- **Worth the cost?** Best method uses 1.0× baseline storage (1026 vs 1026 embeddings) and 0.074ms p95 search, for +0.0337 nDCG@10 over baseline. **Justified** — a real quality gain at modest (≤2×) storage.
