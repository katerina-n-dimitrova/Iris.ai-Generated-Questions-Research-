# Vision-enriched generated-questions experiment — SPIQA test-B (20 papers)

Same multi-vector doc2query experiment as the full run (baseline / q1 / q10 / q50 / q100), on **20 papers · 1026 chunks · 74 eval queries** (74 figure-answerable). **216** figure/table units, **216** enriched with a gpt-4o-mini visual description that is folded into the chunk **before** questions are generated. Embedder huggingface:Octen/Octen-Embedding-0.6B:dim1024; no fine-tuning.

The **vision** arm differs from **novision** only in that figure/table chunks carry a visual-description paragraph, so both the chunk embedding and its doc2query question embeddings describe the actual chart/plot/table content — not just the caption.

## novision arm (baseline replication of the screenshot table)

| condition | n_questions_per_chunk | hit@1 | hit@5 | hit@10 | mrr | ndcg@10 | figQ_hit@5 | figQ_ndcg@10 | figQ_mrr | figQ_n | total_embeddings | index_size_mb | embeddings_x_baseline | search_ms_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.4459 | 0.7568 | 0.8108 | 0.5732 | 0.6131 | 0.7568 | 0.6131 | 0.5732 | 74 | 1026 | 21.251 | 1.0 | 7.026 |
| q1 | 1 | 0.3784 | 0.6757 | 0.8243 | 0.5105 | 0.5715 | 0.6757 | 0.5715 | 0.5105 | 74 | 2052 | 25.226 | 2.0 | 8.741 |
| q10 | 10 | 0.5 | 0.8243 | 0.8919 | 0.6462 | 0.6738 | 0.8243 | 0.6738 | 0.6462 | 74 | 11217 | 72.387 | 10.93 | 11.017 |
| q50 | 50 | 0.527 | 0.7838 | 0.8649 | 0.6399 | 0.6656 | 0.7838 | 0.6656 | 0.6399 | 74 | 17339 | 104.061 | 16.9 | 11.267 |
| q100 | 100 | 0.527 | 0.7838 | 0.8649 | 0.6396 | 0.6648 | 0.7838 | 0.6648 | 0.6396 | 74 | 17538 | 103.866 | 17.09 | 13.068 |

## vision arm (figure chunks vision-enriched before question generation)

| condition | n_questions_per_chunk | hit@1 | hit@5 | hit@10 | mrr | ndcg@10 | figQ_hit@5 | figQ_ndcg@10 | figQ_mrr | figQ_n | total_embeddings | index_size_mb | embeddings_x_baseline | search_ms_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.527 | 0.7568 | 0.7973 | 0.6195 | 0.6407 | 0.7568 | 0.6407 | 0.6195 | 74 | 1026 | 21.883 | 1.0 | 8.445 |
| q1 | 1 | 0.3108 | 0.7027 | 0.8243 | 0.4887 | 0.5638 | 0.7027 | 0.5638 | 0.4887 | 74 | 2052 | 26.175 | 2.0 | 9.027 |
| q10 | 10 | 0.5405 | 0.8514 | 0.9189 | 0.6632 | 0.6911 | 0.8514 | 0.6911 | 0.6632 | 74 | 11235 | 73.067 | 10.95 | 12.992 |
| q50 | 50 | 0.5 | 0.7568 | 0.9054 | 0.6215 | 0.672 | 0.7568 | 0.672 | 0.6215 | 74 | 18825 | 110.608 | 18.35 | 16.807 |
| q100 | 100 | 0.4865 | 0.7568 | 0.8919 | 0.6134 | 0.663 | 0.7568 | 0.663 | 0.6134 | 74 | 19074 | 115.705 | 18.59 | 14.462 |

## Vision effect (vision − novision), per condition

| condition | Δ overall nDCG@10 | Δ overall Hit@5 | Δ figQ nDCG@10 | Δ figQ Hit@5 | Δ figQ MRR |
| --- | --- | --- | --- | --- | --- |
| baseline | +0.0276 | +0.0 | +0.0276 | +0.0 | +0.0463 |
| q1 | -0.0077 | +0.027 | -0.0077 | +0.027 | -0.0218 |
| q10 | +0.0173 | +0.0271 | +0.0173 | +0.0271 | +0.017 |
| q50 | +0.0064 | -0.027 | +0.0064 | -0.027 | -0.0184 |
| q100 | -0.0018 | -0.027 | -0.0018 | -0.027 | -0.0262 |

## Findings

- **All 74 eval queries in this SPIQA test-B sample are figure-answerable** (every SPIQA question refers to a figure/table), so the figure-subset and overall metrics coincide here — vision enrichment potentially affects every query.
- **Vision helps most where it lands directly on the chunk vector:** at `baseline` (no generated questions) it lifts nDCG@10 by **+0.0276** and Hit@1 by **+0.0811** — the chunk embedding now carries the chart/table content, not just the caption.
- **At the best operating point `q10`:** nDCG@10 **+0.0173**, Hit@5 **+0.0271**, Hit@10 **+0.027**, MRR **+0.017** vs the same-count novision arm. Best overall config = vision `q10` (**0.6911** nDCG@10, **0.8514** Hit@5, **0.9189** Hit@10).
- **q1 stays below baseline in both arms** (a single generated question dilutes the chunk vector); **q50/q100 add cost without gain** (question saturation ~20/chunk).
- The vision step added ~216 figure-chunk question-generation calls (~$0.0578) on top of the one-time image-description pass ($1.08, cached & reused); storage/latency track the novision arm (same #questions).

## Recommendation
**Add vision descriptions to figure units before doc2query generation.** It measurably lifts retrieval on this figure-heavy set — most at `baseline`/`q10`, exactly where caption-only retrieval is weakest — at negligible marginal cost. Keep the sweet spot at **q10** (best quality/storage trade-off; q50/q100 saturate).
