# Memory index

- [SPIQA num-questions experiment](spiqa-num-questions-experiment.md) — new sub-project under src/spiqa/: how #generated-questions/chunk affects retrieval vs latency.
- [SPIQA dataset schema](spiqa-dataset-schema.md) — the three small SPIQA splits use different JSON schemas; only test-B/test-C have body text.
- [Statement q10 findings](spiqa-statement-q10-findings.md) — statement-level score fusion is the first method to clearly beat baseline on SPIQA test-B; filtering/rerank hurt.
- [SPIQA follow-up question quality](spiqa-followup-question-quality.md) — quality>count study; structured figure/table context wins (0.7134), diverse/bm25-aware/separate-index/reranker don't beat q10 hybrid.
- [PeerQA dataset schema](peerqa-dataset-schema.md) — PeerQA is pre-chunked at sentence level; gold = sentence-idx sets; only 90/208 papers have redistributable text (rest via arXiv fetch).
- [PeerQA questions-only experiment](peerqa-questions-only-experiment.md) — src/peerqa/: index embeds ONLY generated questions (no chunk text); wins recall, loses early precision, monotonic in count.
- [PeerQA chunk-size experiment](peerqa-chunksize-experiment.md) — chunk size × question count; bigger chunks retrieve better & want more questions, plain 800t chunk-text beats questions-only, adaptive length-based beats fixed q10.
- [DocBank enrichment experiment](docbank-enrichment-experiment.md) — src/docbank/: layout-heavy arXiv docs; questions-only BEATS chunk-text at all counts (q13 best), but synthetic-eval/enrichment share LLM origin (caveat).
- [DocBank dataset access](docbank-dataset-access.md) — sample DocBank without the 500K-page download via remotezip range-reads on the txt zip (filenames carry arXiv doc id + page).
- [QASPER baseline harness](qasper-baseline-harness.md) — src/qasper/: which question-TYPE to generate (dense+BM25+RRF); B0/B1 validated, naive questions-only loses early precision (reproduces peerqa); 5 experiments pending.
- [MultiHop-RAG baseline harness](multihoprag-baseline-harness.md) — src/multihoprag/: same question-KIND study on MultiHop-RAG news (multi-evidence, Evidence Recall@k); B0/B1 validated on 10-article pilot (69q), naive B1 ~ties B0 (n.s.); 5 experiments pending.
- [MultiHop-RAG vector-only pilot](mhrag-vectoronly-pilot.md) — src/mhrag_vectoronly/: DENSE-only (no BM25/hybrid) 15-article pilot, chunk vectors vs 10-questions/chunk; B does NOT beat A (tie ER@5, A wins Hit@4/5); HTML report.
- [MultiHop-RAG atomic+chunk-level mix pilot](mhrag-atomic-chunk-mix-pilot.md) — src/mhrag_atomic_mix/: 10-article dense-only pilot, chunk vectors (A) vs pooled atomic+chunk-level filtered questions (E); E SIGNIFICANTLY HURTS (ER@5 0.489→0.349, p=0.001); HTML report.
