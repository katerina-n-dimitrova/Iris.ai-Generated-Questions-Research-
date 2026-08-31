# QASPER — question-generation strategy study

Which *kind* of LLM-generated question best enriches a chunk for retrieval?
For every paragraph-chunk of a scientific paper an LLM generates synthetic
questions the chunk can answer (doc2query-style). The questions are embedded and
indexed; a user query is matched against them and routed back to the parent
chunk. This sub-project isolates *what to generate* on QASPER (QA over NLP
papers), with a hybrid **dense + BM25 + RRF** retriever.

> **Status:** ships the **baselines (B0, B1)** and **Experiment 1 (semantic
> question type, arm `E1`)**, end-to-end. Experiments 2–5 plug into the same
> data / indexing / retrieval / evaluation modules by adding a prompt + an `Arm`.
>
> **Corpus (`SELECTION_MODE=text_only`, default):** QASPER papers are float-heavy,
> so to study generated questions **on text only** we select papers that are pure
> text — **zero figures/tables/charts, formula-free, ≥4 answerable text questions**
> — pooled over train+dev. Only ~4 QASPER papers qualify (306 chunks, 18 queries),
> so **results are directional only: at n=18 the 95% CIs overlap heavily.** Broaden
> by relaxing `QASPER_MAX_FLOATS` / `QASPER_MIN_QUERIES`, or set
> `QASPER_SELECTION=random_dev QASPER_NUM_PAPERS=15` for the larger mixed pilot.

## The fixed factors (identical across every arm)

| Factor | Setting |
|---|---|
| Chunking | one paragraph = one chunk (QASPER evidence is paragraph-level) |
| Budget | exactly 10 generated questions / chunk |
| Generator | `gpt-4o-mini` @ temp 0.3 (one model + temp for all arms; only the prompt changes) |
| Embedder | `Octen/Octen-Embedding-0.6B` (repo default; never changed) |
| **Dense index** | **only generated-question vectors** — each question one vector → its parent chunk; a chunk's dense score is the **max** over its own questions. The chunk text is **never** embedded in an enrichment arm. |
| Chunk embeddings | exist in **exactly two** arms: **B0** (no questions) and the future **Exp‑4 variant (f)** (questions+chunk, built to test whether dropping the chunk vector is right). Enforced by `qasper_config.validate_arm`. |
| Sparse | BM25 over chunk text, with generated questions appended to the BM25 document in enrichment arms |
| Fusion | Reciprocal Rank Fusion of dense + BM25, k=60 |

**Baselines** — `B0`: no enrichment (plain chunk vector + BM25 chunk text).
`B1`: naive enrichment ("generate 10 questions this passage answers").

## Setup

Uses the parent project's `.venv` and `.env` (OpenAI key, embedder config).

```bash
cd context-enrichment-rag
source .venv/bin/activate          # Python 3.10, deps already installed
# one-time: fetch QASPER v0.3 (train+dev) into data/raw/qasper/
mkdir -p data/raw/qasper && cd data/raw/qasper
curl -sL -o qasper.tgz https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
tar xzf qasper.tgz && cd -
```

## Run it (one command per stage)

```bash
cd src/qasper
python qasper_data.py                       # build corpus + queries + gold (cached)
python run_qasper.py                        # B0 + B1 + E1 + Exp-1 cross-tab + report
python qasper_report.py                     # rebuild results.md + results.html from saved metrics
```

`run_qasper.py` is the umbrella runner (data → generation → indexing → retrieval →
evaluation → Experiment-1 analysis → report). Useful flags:

```bash
python run_qasper.py --arms B0 B1                # baselines only
python run_qasper.py --skip-generate            # reuse cached questions
python run_qasper.py --eval-only                # recompute metrics + report from saved rankings
```

(`run_qasper_baselines.py` is the older baseline-only runner; `run_qasper.py`
supersedes it.)

Everything is cached: generated questions (`data/processed/qasper/generated/`),
and per-query rankings (`results/qasper/rankings/`). Re-runs never regenerate or
re-embed unless the cache is missing.

## Metrics

Recall@1/5/10, MRR@10, nDCG@10 — reported **overall**, **per QASPER answer type**
(extractive / abstractive / boolean), and **per retrieval mode** (dense / BM25 /
hybrid) so you can see *where* gains come from. 95% confidence intervals come
from a 1,000-resample bootstrap over queries; arm-vs-arm significance uses a
paired bootstrap. All rankings are saved so metrics recompute without retrieval.

## Scaling to the full split

Change **one** value — the pilot's 15 papers → the whole dev split:

```bash
QASPER_NUM_PAPERS=0 python qasper_data.py   # 0 (or ≥ split size) = all dev papers
QASPER_NUM_PAPERS=0 python run_qasper_baselines.py
```

Paper selection is seeded (`QASPER_SEED=42`) and saved to
`data/processed/qasper/selected_papers.json`.

## Layout

```
src/qasper/
  qasper_config.py      # fixed factors, arm registry + invariant guard, paths, seeds
  qasper_data.py        # load QASPER, seeded 15-paper sample, chunks, queries, gold labels
  qasper_generate.py    # cached/resumable question generation; prompt registry; failure logging
  qasper_index.py       # dense (Chroma, questions-only / chunk-only) + BM25 index builders
  qasper_retrieval.py   # dense / BM25 rankings + RRF fusion; saves rankings to disk
  qasper_eval.py        # Recall/MRR/nDCG + bootstrap CIs + paired significance
  qasper_report.py      # results.md + self-contained results.html (rebuildable)
  run_qasper_baselines.py  # orchestrator: B0 + B1 end-to-end

data/processed/qasper/  # selected_papers.json, chunks.jsonl, queries.jsonl, generated/
results/qasper/         # metrics_<arm>.json, rankings/, report/results.{md,html}, run summary
chroma_indexes/qasper/  # per-arm dense collections
```

## Adding an experiment arm (experiments 1–5)

1. Add a prompt to `qasper_generate.PROMPTS`.
2. Register an `Arm` in `qasper_config.ARMS` (`embeds_chunk=False` for every
   enrichment arm except the Exp‑4 variant `E4f`; `validate_arm` enforces this).
3. `run_qasper_baselines.py --arms <name>` — indexing / retrieval / evaluation
   and the report need no changes.

## Findings so far

See `results/qasper/report/results.{md,html}` (rebuildable via `qasper_report.py`).

- **15-paper mixed pilot (`random_dev`, superseded as the main corpus):** naive
  enrichment (B1) is *significantly worse* than plain chunks (B0) on early precision
  (Recall@1/5, MRR@10, nDCG@10) while matching Recall@10 and beating it on deep recall
  (Recall@100 0.686 vs 0.643) — the questions-only trade, reproducing the peerqa
  finding. Experiment 1 (E1, type-stratified) clearly beat B1 there.
- **4-paper text-only run (current, `text_only`):** with only 18 queries the arms are
  **statistically indistinguishable** (95% CIs overlap). Directionally B0 ≥ B1 ≈ E1;
  E1 is significantly worse than B1 on MRR@10 only. Same-type-on-gold coverage rises
  B1 94% → E1 100%, and naive B1 over-produces CONCEPT (~29%), but the cross-tab
  'absent' cell collapses to n=1, so the mechanism can't be tested at this scale.
- **Index integrity verified** in both runs (chunk→self rank 1; generated question→
  parent rank 1; no gold id missing from the corpus).
- **Recommendation:** the strict text-only constraint is too restrictive on QASPER
  (~4 qualifying papers). To actually separate question-type strategies, broaden the
  text-leaning sample (relax the float/query thresholds or use text-answerable
  questions over more papers).
