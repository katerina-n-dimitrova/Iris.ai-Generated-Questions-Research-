# Yettel Bulgaria — five adaptive-question conditions

Self-built Bulgarian corpus: 340 Yettel.bg documents, 963 chunks (1024/128),
2,556 generated evaluation queries (2,255 evidence-bearing + 301 null, the
nulls excluded from retrieval metrics). Generator `gpt-5.4-mini`, Iris dim-384
embeddings, Unicode Bulgarian BM25, RRF k=60.

Verified MRR@10: 0.392 (baseline) → 0.513 (5–20) → 0.516 (unbounded) →
0.547 (chunk + article 5–20) → 0.548 (chunk + article unbounded).

## Running the experiments

```bash
python exp/yettel_bg/run_experiments.py
```

One run covers all five conditions and renders
`report/yettel_bg_adaptive_questions.html`. The runner reuses
`exp/multihoprag/adaptive_lib.py` and `chunk_article_questions.py`, repointing
their module-level paths/prompts at the Bulgarian corpus (Bulgarian fact and
question prompts, Unicode BM25 tokenizer) before calling them.

## Building the dataset (`corpus/`)

Only needed to rebuild `data/processed/yettel_bg/` from scratch:

```bash
python exp/yettel_bg/corpus/build_corpus.py        # crawl sitemap -> documents + chunks
python exp/yettel_bg/corpus/generate_questions.py  # build the 2,556-query benchmark
python exp/yettel_bg/corpus/validate_dataset.py    # integrity checks
```

The evaluation questions and the enrichment questions use separate prompts,
seeds, and artifacts, so there is no direct leakage — but both were generated
by `gpt-5.4-mini`, so describe results as a controlled same-model synthetic
benchmark.
