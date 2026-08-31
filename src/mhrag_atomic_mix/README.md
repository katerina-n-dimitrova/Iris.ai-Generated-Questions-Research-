# MultiHop-RAG — 10-article atomic + chunk-level mixed-question pilot

A closed-collection, 10-article MultiHop-RAG **dense-vector** pilot comparing
original-chunk embeddings with a **pooled mixture of atomic-fact and chunk-level
synthetic-question** embeddings. Vector search only — no BM25/sparse/hybrid/rerank.

* **Condition A** (`baseline`): one vector per original chunk.
* **Condition E** (`generated`): per chunk → decompose into self-contained atomic
  facts → 1 focused question per important atom (+ optional 2nd view) → 2–3 broader
  chunk-level questions → **validate + filter** (grounding, self-containedness,
  near-dup, round-trip parent-in-top-3, confusion-margin, coverage-aware) → pool
  accepted questions as separate vectors linked to the parent chunk; chunk score =
  max cosine over its questions.

Held fixed across A/E: 10 articles, cleaning, 256/50/80 chunks, eligible queries,
embedding model (Octen-Embedding-0.6B, cosine), k=[1,3,4,5,10], gold mapping,
answer LLM + prompt, seed 42. Reuses `src/mhrag_vectoronly/` for cleaning,
tokenizer, gold alignment, the metric suite, and answer scorers.

## Run

```bash
cd src/mhrag_atomic_mix
python run_am.py                 # whole pipeline (cached/resumable)
python run_am.py --stage prepare|generate|filter|index|retrieve|
                 evaluate_retrieval|diagnostics|evaluate_generation|
                 analyze_failures|report
python run_am.py --stage generate --force  # deliberately replace that cache
```

Config: `config/mhrag_atomic_chunk_mix_10.yaml`. Credentials only from `.env`.
Local Chroma namespace `mhrag_atomic_chunk_mix_10` (isolated from the 15-article run).

## Result (gpt-5.4-mini run, n=69 eligible queries)

**Roughly a statistical tie.** Evidence Recall@5 was 0.489 for A versus 0.475
for E (Δ−0.015, paired bootstrap p=0.700). Condition E therefore matched the
plain-chunk baseline on the primary metric, but did not beat it, while indexing
676 rather than 96 vectors (7.04×). See the generated HTML report for the full
metric sweep, answer scores, and failure analysis.

> A closed-collection, 10-article pilot — NOT the full-corpus MultiHop-RAG benchmark.
