# Statement-level q10 generated-question enrichment — summary

SPIQA **test-B**. Generated questions are statement-level, anchor-heavy, type-tagged, and used only as a **secondary** signal that maps back to the parent chunk (never independent evidence). Same LLM + same embedding model (`huggingface:Octen/Octen-Embedding-0.6B:dim1024`) across all conditions. No fine-tuning.

## Comparison table

| condition | hit@1 | hit@5 | hit@10 | mrr | ndcg@10 | generated_questions | kept_questions | filtered_pct | embeddings | index_mb | storage_x | search_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.4459 | 0.7568 | 0.8108 | 0.5696 | 0.6128 | 0 | 0 | 0.0 | 1026 | 9.328 | 1.0 | 0.091 |
| q10_statement_raw_append | 0.5 | 0.7703 | 0.8514 | 0.6169 | 0.6437 | 4911 | 4911 | 0.0 | 1026 | 9.352 | 1.0 | 0.138 |
| q10_statement_roundtrip_filter_append | 0.4595 | 0.7703 | 0.8243 | 0.5947 | 0.6344 | 4911 | 2928 | 40.4 | 1026 | 9.352 | 1.0 | 0.141 |
| q10_statement_score_fusion | 0.5811 | 0.7568 | 0.8514 | 0.6647 | 0.6748 | 4911 | 4911 | 0.0 | 5937 | 35.826 | 3.84 | 0.521 |
| q10_statement_filtered_score_fusion | 0.527 | 0.7703 | 0.8378 | 0.6281 | 0.6534 | 4911 | 2928 | 40.4 | 3954 | 26.815 | 2.87 | 0.537 |
| q10_statement_filtered_fusion_rerank | 0.5 | 0.7432 | 0.8514 | 0.5998 | 0.638 | 4911 | 2928 | 40.4 | 3954 | 26.815 | 2.87 | 0.374 |

## Findings

- **Best generated-question condition:** `q10_statement_score_fusion` (nDCG@10 0.6748, +0.062 vs baseline).
- **Did statement-level questions help retrieval?** Yes — the best condition beats baseline on nDCG@10.
- **Round-trip + anchor filtering:** did not help (-0.0093 nDCG@10 for the append variant; 40.4% removed).
- **Score fusion (0.7·orig + 0.3·best-question):** helped (+0.0311 nDCG@10 vs plain append).
- **Rerank top-20 by original chunk text:** did not help (-0.0154 nDCG@10 vs fusion without rerank).
- **Anchors:** anchored questions round-trip to their own chunk 68% of the time vs 64% for non-anchored → anchor-heavy questions DO retrieve better.
- **Worth the storage/latency?** `q10_statement_score_fusion` uses 3.84× baseline storage (5937 vs 1026 embeddings), 0.521ms p95, for +0.062 nDCG@10. **Justified** at this modest overhead.

See `question_type_analysis.json`, `anchor_analysis.json`, `duplicate_question_analysis.json`, `filtered_questions_examples.json`, and `improved/hurt/unchanged_queries.json` for detail.
