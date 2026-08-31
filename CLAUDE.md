# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The final study on **context enrichment in RAG retrieval**: does indexing
LLM-generated questions (doc2query / HyPE style) alongside raw chunk vectors
improve retrieval? Five controlled conditions (baseline, adaptive chunk
questions 5–20 / unbounded, chunk + whole-article questions 5–20 / unbounded)
on two corpora — MultiHop-RAG (609 English news articles) and a self-built
Yettel Bulgaria telco corpus (340 Bulgarian documents) — plus a union-corpus
interference study and a Qwen-generator replication. `README.md` holds the
condition table and verified numbers.

It is an **experiment repo, not a library or a product**. No test suite, no
CI — the smoke run is the test. Deliverables are the self-contained HTML files
in `report/`.

Note: the parent directory name ends in a **trailing space**
(`.../Structured, Unstructured, Tables, Charts Context Enrichment /`). Always
quote paths; `cd` into a bare unquoted path will fail.

## Layout

- `src/ragkit/` — the shared package (installed by `uv sync`): `config`
  (.env + OpenAI client), `embeddings` (Iris / OpenAI / HF backends), `text`
  (ASCII + Unicode BM25 tokenizers, JSONL I/O), `vectors` (cosine helpers),
  `fusion` (RRF k=60), `metrics` (evidence recall / MRR@10).
- `exp/multihoprag/` — the five conditions on MultiHop-RAG. `run.py` enforces
  the strict order (the scripts share one metrics.json and one report):
  `baseline` → `adaptive_questions` (conditions 1–3) →
  `chunk_article_questions` (appends 4) → `chunk_article_unbounded` (appends 5,
  renders the final report). `adaptive_lib.py` is the shared engine.
- `exp/yettel_bg/` — the same five conditions in one runner, plus `corpus/`
  (sitemap crawler, benchmark question generator, validator).
- `exp/combined/` — union-corpus study. Its per-dataset numbers are NOT the
  headline numbers (mhrag baseline 0.641 here vs 0.644 standalone).
- `exp/qwen_generator/` — conditions 1/2/4 with the Iris-hosted Qwen3.5-4B
  generator; the Yettel half is unfinished.

## The configuration-by-monkey-patching convention

`exp/multihoprag/adaptive_lib.py` exposes all knobs (paths, prompts, budgets,
`call_json`, `generate_one`, `tokenize`, `cosine_scores`, `embed_resumable`)
as **module attributes**, and every other experiment configures it by
assignment before calling into it:

- `adaptive_questions.py` repoints the paths at the full corpus **at import
  time** (importing it has side effects, and `chunk_article_questions.py`
  imports it precisely for those side effects).
- `exp/yettel_bg/run_experiments.py` swaps in Bulgarian prompts, the Unicode
  tokenizer, and Yettel paths.
- `exp/qwen_generator/*` swaps `call_json`/`generate_one` for the Iris Qwen
  endpoint and isolated caches.

Consequences: keep helper functions imported *into* module namespaces (`from
ragkit.text import tokenize_ascii as tokenize`) rather than referenced via
their home module, or the rebinding breaks; never rename `adaptive_lib`
attributes without checking all four consumer experiments; import order in
those consumers is load-bearing.

## Environment

```bash
./prep.sh                # uv venv + uv sync (installs ragkit editable)
cp .env.example .env     # fill in OPENAI_API_KEY (+ IRIS_* for iris backend)
```

Config and credentials come **only** from the repo-root `.env`, read by
`ragkit/config.py` and `ragkit/embeddings.py`. All presentation runs used
`EMBEDDING_BACKEND=iris` (dim-384) and `OPENAI_CHAT_MODEL=gpt-5.4-mini`.
The HF backend needs `uv sync --extra hf`.

## Caches and experimental discipline

- **Every stage is cached and resumable.** Re-running never regenerates
  questions or re-embeds unless the cache is missing or its text list
  changed. The LLM-generation caches under `data/processed/<experiment>/`
  (`adaptive_generations.jsonl`, `article_question_generations*.jsonl`) are
  costly artifacts, not scratch. Vector caches live in `results/<experiment>/`
  (gitignored `*vector*.json`); with them present, a full re-evaluation is
  offline and byte-reproducible.
- `data/`, vector caches, and `.env` are gitignored; metrics, rankings,
  reports, and code are tracked. Never assume a data file exists on a fresh
  clone.
- **Only the enrichment changes between conditions** — chunks, embedder,
  BM25, RRF k, seeds (42) are held fixed. The ASCII vs Unicode BM25 tokenizer
  split (`ragkit/text.py`) is deliberate per-corpus; swapping one for the
  other silently shifts BM25 scores.
- Before an expensive run, smoke-test against the shipped caches: a full
  offline re-run of `exp/yettel_bg/run_experiments.py` reproduces
  `results/yettel_bg_experiments/metrics.json` byte-for-byte.
