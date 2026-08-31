# MultiHop-RAG — question-generation strategy study

Which *kind* of LLM-generated question best enriches a chunk for retrieval? For
every paragraph-chunk of a news article an LLM generates synthetic questions the
chunk can answer (doc2query / HyPE-style). The questions are embedded and
indexed; a user query is matched against them and routed back to the parent
chunk. This sub-project isolates *what to generate* on **MultiHop-RAG** (Tang &
Yang 2024), with a hybrid **dense + BM25 + RRF** retriever.

> **Status:** ships the **baselines (B0, B1)** end-to-end. Experiments 1–5 plug
> into the same data / generation / indexing / retrieval / evaluation / report
> modules by adding a prompt + an `Arm` — no changes to the harness.

## Fixed factors (identical across every arm)

| Factor | Setting |
|---|---|
| Chunking | article body split on blank lines into paragraph chunks; paragraphs `< MERGE_MIN_TOKENS` (40) merged with a neighbour |
| Budget | exactly 10 generated questions / chunk |
| Generator | `gpt-4o-mini` @ temp 0.3 (one model + temp for all arms; only the prompt changes) |
| Embedder | `Octen/Octen-Embedding-0.6B` (repo default; never changed) |
| **Dense index** | **only generated-question vectors** — each question one vector → its parent chunk; a chunk's dense score is the **max** over its own questions. Chunk text is **never** embedded in an enrichment arm. |
| Chunk embeddings | exist only in **B0** and the Experiment-4 variants **(b)/(e)/(f)**. Enforced by `mhrag_config.validate_arm`. |
| Sparse | BM25 over chunk text, with generated questions appended to the BM25 document in enrichment arms (Exp-4 varies this) |
| Fusion | Reciprocal Rank Fusion of dense + BM25, k = 60 |

**Baselines** — `B0`: no enrichment (plain chunk vector + BM25 chunk text).
`B1`: naive enrichment ("generate 10 questions this passage answers").

## Dataset (query-first pilot selection)

MultiHop-RAG queries are multi-evidence (each cites 2–4 articles), so random
article samples leave almost no answerable query. Selection is **query-first**
(`mhrag_data.select_articles_and_queries`): drop null queries → shuffle (seed 42)
→ greedily accept a query iff its evidence articles fit within `ARTICLE_BUDGET`
→ then keep every remaining query whose evidence articles are all selected.
Pilot (`ARTICLE_BUDGET=10`): **10 articles → 180 chunks, 69 queries** (inference
29 / temporal 23 / comparison 17), avg 2.71 gold chunks/query. All 187 evidence
facts land in exactly one chunk (0 unmatched — logged in `dataset_summary.json`).

Gold label: a chunk is relevant to a query iff it contains one of the query's
evidence `fact` snippets (normalised-whitespace substring, else fuzzy ≥ 0.9).

## Setup

Uses the parent project's `.venv` and `.env` (OpenAI key, embedder config).

```bash
cd context-enrichment-rag
source .venv/bin/activate
# one-time: fetch the dataset into data/raw/multihoprag/
python - <<'PY'
from huggingface_hub import hf_hub_download; import shutil, os
os.makedirs("data/raw/multihoprag", exist_ok=True)
for fn in ["corpus.json","MultiHopRAG.json"]:
    shutil.copy(hf_hub_download("yixuantt/MultiHopRAG", fn, repo_type="dataset"),
                f"data/raw/multihoprag/{fn}")
PY
```

## Run it (one command per stage)

```bash
cd src/multihoprag
python mhrag_data.py                    # build corpus + queries + gold (cached)
python run_mhrag.py                     # B0 + B1 end-to-end + report
python run_mhrag.py --eval-only         # recompute metrics + report from saved rankings
python mhrag_report.py --arms B0 B1     # rebuild results.md + results.html only
```

Everything is cached: generated questions (`data/processed/multihoprag/generated/`)
and per-query rankings (`results/multihoprag/rankings/`). Re-runs never
regenerate or re-embed unless the cache is missing.

## Metrics

Queries are multi-evidence, so the **primary metric is Evidence Recall@k** =
fraction of a query's gold evidence chunks in the top k (k = 2, 5, 10). Also:
**Hit@k** (≥1 gold in top k), **MRR@10**, **nDCG@10**. Reported **overall**, per
**MultiHop-RAG query type** (inference / comparison / temporal), and per
**retrieval mode** (dense / BM25 / hybrid) so gains can be localised. 95% CIs
from a 1,000-resample bootstrap over queries; arm-vs-arm significance uses a
paired bootstrap. All rankings are saved so metrics recompute without retrieval.

## Scaling to more articles

Change **one** value — the pilot's 10 articles → 50/100 for the confirmation run:

```bash
MHRAG_ARTICLE_BUDGET=50 python mhrag_data.py
MHRAG_ARTICLE_BUDGET=50 python run_mhrag.py
```

Selection is seeded (`MHRAG_SEED=42`) and saved to
`data/processed/multihoprag/selected_articles.json` + `selected_query_ids.json`.

## Layout

```
src/multihoprag/
  mhrag_config.py     # fixed factors, arm registry + invariant guard, paths, seeds
  mhrag_data.py       # load MultiHopRAG, query-first selection, chunking, gold labels
  mhrag_generate.py   # cached/resumable question generation; prompt registry; failure log
  mhrag_index.py      # dense (Chroma, questions-only / chunk-only) + BM25 index builders
  mhrag_retrieval.py  # dense / BM25 rankings + RRF fusion; saves rankings to disk
  mhrag_eval.py       # Evidence Recall / Hit / MRR / nDCG + bootstrap CIs + paired significance
  mhrag_report.py     # results.md + self-contained results.html (rebuildable)
  run_mhrag.py        # umbrella runner

data/processed/multihoprag/  # selected_articles.json, selected_query_ids.json, chunks.jsonl, queries.jsonl, generated/
results/multihoprag/         # metrics_<arm>.json, rankings/, report/results.{md,html}
chroma_indexes/multihoprag/  # per-arm dense collections
```

## Adding an experiment arm (experiments 1–5)

1. Add a prompt to `mhrag_generate.PROMPTS`.
2. Register an `Arm` in `mhrag_config.ARMS` (`embeds_chunk=False` for every
   enrichment arm except the Exp-4 variants `E4b`/`E4e`/`E4f`; `validate_arm`
   enforces this).
3. `run_mhrag.py --arms <name>` — indexing / retrieval / evaluation / report
   need no changes.

## Baseline finding (pilot, validated harness)

At 10 articles / 69 queries, **naive enrichment (B1) does not beat plain chunks
(B0)** — directionally B1 is slightly *behind* on early precision (hybrid
Evidence Recall@2 0.374 vs 0.406, MRR@10 0.716 vs 0.768) but **no metric is
significant** (paired bootstrap CIs include 0; e.g. ER@5 Δ = −0.021,
p = 0.47). The mode breakdown localises it: B1's **questions-only dense index is
worse than B0's chunk vectors** (dense MRR 0.586 vs 0.699), while **appending
questions to BM25 helps sparse** (BM25 ER@5 0.610 vs 0.556); hybrid RRF nets out
roughly even. This reproduces the QASPER/PeerQA "questions-only loses early
precision, BM25-append helps" pattern and motivates Experiments 3 (implicit /
vocab-bridging) and 5 (style-match) as the likely fixes for the dense deficit.
Integrity verified: chunk→self rank 1, generated-question→parent rank 1, 0 gold
chunks missing. Corpus is tiny (Hit@10 = 1.0), so treat as a pilot signal.
```
