# SPIQA — Number of Generated Questions per Chunk (retrieval quality vs latency)

**Research question:** *How does the number of generated questions per chunk
affect retrieval quality and latency in a RAG system?*

Conditions compared on SPIQA:

| Condition | Questions/chunk | Collection |
|-----------|-----------------|------------|
| baseline  | 0  | `spiqa_testB_baseline` |
| q1        | 1  | `spiqa_testB_q1` |
| q10       | 10 | `spiqa_testB_q10` |
| q50       | 50 | `spiqa_testB_q50` |
| q100      | 100| `spiqa_testB_q100` |

## Indexing strategy (multi-vector doc2query)

* The **original chunk** is always stored with its own embedding.
* Each **generated question** is stored as its **own** embedding, linked to the
  parent via `metadata.parent_chunk_id`.
* At retrieval, a question hit is **mapped back to its parent chunk**, and the
  **chunk text is the returned evidence** — never the question.
* Each condition lives in its **own on-disk Chroma persist directory**
  (`chroma_indexes/spiqa/<collection>/`) so per-condition **storage size** is
  measurable (Stage 8). Persistence is forced **local** even if the shared
  `.env` sets `CHROMA_MODE=cloud`.

## Modules (all parameters configurable via CLI flags / `SPIQA_*` env vars)

| File | Role |
|------|------|
| `spiqa_config.py` | Paths, split registry, chunk/overlap/top-k/Q-condition knobs, collection naming, pricing. Reuses the repo's `config.py` (dotenv, paths). |
| `llm_adapter.py`  | OpenAI-compatible LLM (`openai` default, optional `litellm`). **API key only from `OPENAI_API_KEY`** — never hard-coded. Returns `(text, usage)` for cost tracking. |
| `spiqa_loader.py` | Downloads only the **small** split JSONs (never the 208 MB train JSON / multi-GB image zips); normalises SPIQA's three schemas into a unified `Paper` model; saves inspection samples. |
| `spiqa_chunker.py`| Token-based (tiktoken) chunking, default **416 tokens / 100 overlap**, section-aware. Figures/tables become their **own retrieval units** (caption + metadata + nearby context + heading). Rich metadata per chunk. |
| `question_gen.py` | Generates up to `MAX_QUESTIONS` grounded questions/chunk (no unsupported facts). **Resumable disk cache**; the smaller conditions are the **first-k subset** of the max pool (isolates *count*, not prompt; ~1 call/chunk). Tracks time + tokens + cost. |
| `spiqa_index.py`  | Builds all condition collections. Embeds chunks + the 100-question pool **once**, slices for smaller conditions. Tracks record counts, add time, index size. |
| `spiqa_eval.py`   | Builds gold from SPIQA (`referred_figures_tables` = exact figure/table gold; `evidential_info` verbatim excerpts → matched text chunks; **paper-level fallback** documented). Over-fetch + map-to-parent retrieval. Metrics: Hit@1/5/10, Recall@k, MRR, nDCG@10 + query latency (mean/p50/p95). |
| `run_experiment.py` | Orchestrates Stages 1–8; `check-env` and `run` subcommands. |

## SPIQA schema notes (important)

The three small splits use **different** JSON schemas:

* **test-B** (65 papers, **default**): richest — `passages[]` (body text),
  `all_figures_tables{}` captions, and per-question `evidential_info` +
  `referred_figures_tables` (gold). Best for text-retrieval eval.
* **test-C** (314 papers): `full_text[]` (sectioned) + `figures_and_tables[]` +
  `question/answer/referred_figures_tables`. Also fully supported.
* **test-A / val** (118 / 200 papers): image-QA — figure captions only, **no
  body text** and no per-question gold figure ids. Supported as caption-only.

## Running

```bash
cd src/spiqa
python run_experiment.py check-env --live-llm         # Stage 1
python run_experiment.py run --split test-B --max-papers 4   # cheap smoke
python run_experiment.py run --split test-B           # full test-B
# reuse cached questions/indexes, re-eval only:
python run_experiment.py run --split test-B --skip-generation --skip-index
```

Key flags: `--split --max-papers --chunk-size --overlap --top-k --conditions
--gen-limit --skip-generation --skip-index`. Env knobs: `SPIQA_LLM_MODEL`,
`SPIQA_LLM_BACKEND` (openai|litellm), `SPIQA_CHUNK_SIZE`, `SPIQA_CHUNK_OVERLAP`,
`SPIQA_Q_CONDITIONS`, `SPIQA_TOP_K`, `EMBEDDING_BACKEND`, `HF_EMBEDDING_MODEL`.

## Outputs (`results/spiqa/`)

* `test-B_results_table.csv` / `.md` — the Stage 8 comparison table.
* `test-B_results.json` — full generation + index + eval detail.
* `test-B_index_summary.json` — per-condition record counts + sizes.
* `data/processed/spiqa/test-B_chunks.jsonl` — chunks + metadata.
* `data/processed/spiqa/test-B_questions.jsonl` — generated-question cache.
* `data/processed/spiqa/test-B_samples.json` — normalised examples to inspect.

## Known finding baked into the design

When asked for 100 questions, the LLM produces only ~15–20 *distinct* grounded
questions for a typical ~300-token chunk (it saturates). Consequently **q50 and
q100 are nearly identical** (few chunks support >50 questions) — a diminishing-
returns result the experiment surfaces directly.
