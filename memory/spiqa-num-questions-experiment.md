---
name: spiqa-num-questions-experiment
description: New SPIQA sub-project (src/spiqa/) studying how #generated-questions per chunk affects retrieval quality vs latency.
metadata:
  type: project
---

Research question: "How does the number of generated questions per chunk affect
retrieval quality and latency in a RAG system?" Built as a self-contained package
`src/spiqa/` (loader, chunker, question_gen, index, eval, run_experiment) reusing
the repo's `config.py`/`embeddings.py`. Distinct from the existing enrichment work
which embeds *one* enriched vector per chunk — this uses a **multi-vector doc2query**
index: each generated question is its own embedding linking back to `parent_chunk_id`,
and the original chunk is the returned evidence.

Conditions: baseline(0)/q1/q10/q50/q100 questions per chunk. Default split **test-B**
(65 papers). Each condition gets its own on-disk Chroma dir under
`chroma_indexes/spiqa/` so per-condition storage size is measurable (forces LOCAL
chroma even though the shared .env sets CHROMA_MODE=cloud).

**Why:** the user's internship research question; needed a new indexing approach the
existing codebase didn't have.

**How to apply:** run `python run_experiment.py run --split test-B` from `src/spiqa/`.
Questions are generated once at the max (100) and cached (resumable) in
`data/processed/spiqa/test-B_questions.jsonl`; smaller conditions are first-k subsets.
Key finding: LLM saturates at ~15-20 distinct grounded questions/chunk, so q50≈q100.
Cost ~$0.80 / ~40 min wall for full test-B on gpt-4o-mini. See [[spiqa-dataset-schema]].
