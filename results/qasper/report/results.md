# QASPER question-generation study — results

_Regenerated from saved metrics/rankings; no retrieval is re-run._

## Run configuration

- Embedder: `huggingface:Octen/Octen-Embedding-0.6B:dim1024`
- Generator: `gpt-4o-mini` @ temp 0.3, budget 10 questions/chunk
- Papers: 10 (seed 42) · Chunks: 321 · Queries: 44
- Answer types: {'extractive': 38, 'boolean': 5, 'abstractive': 1}
- Fusion: RRF k=60 · CIs: 95% bootstrap, 1000 resamples

## Arms

- **B0** — No enrichment: plain chunk vector + BM25(chunk text) + RRF.
- **B1** — Naive enrichment: 'generate 10 questions this passage answers' (questions-only dense index + BM25 chunk+questions).
- **E1** — Type-stratified: 10 questions covering distinct Cao&Wang semantic types (JUDGMENTAL skipped), allocated to the real-query type distribution.

## Overall — hybrid (dense + BM25, RRF)

Mean [95% bootstrap CI]. **Bold** = best arm per metric. †/‡ = significantly better/worse than **B1** (paired bootstrap).

| metric | B0 | B1 | E1 |
|---|---|---|---|
| recall@1 | **0.184 [0.090, 0.288]** | 0.157 [0.072, 0.252] | 0.184 [0.091, 0.282] |
| recall@5 | 0.432 [0.309, 0.559] | **0.479 [0.348, 0.602]** | 0.433 [0.312, 0.556] |
| recall@10 | **0.599 [0.468, 0.724]** | 0.572 [0.426, 0.705] | 0.596 [0.467, 0.719] |
| mrr@10 | 0.402 [0.292, 0.521] | 0.369 [0.265, 0.479] | **0.409 [0.296, 0.538]** |
| ndcg@10 | **0.420 [0.320, 0.524]** | 0.397 [0.298, 0.499] | 0.412 [0.309, 0.514] |

## Where gains come from — dense / BM25 / hybrid

### recall@10

| mode | B0 | B1 | E1 |
|---|---|---|---|
| dense | **0.593 [0.458, 0.730]** | 0.579 [0.455, 0.699] | 0.575 [0.449, 0.702] |
| bm25 | 0.454 [0.309, 0.585] | **0.550 [0.420, 0.669]** | 0.510 [0.371, 0.636] |
| hybrid | **0.599 [0.468, 0.724]** | 0.572 [0.426, 0.705] | 0.596 [0.467, 0.719] |

### mrr@10

| mode | B0 | B1 | E1 |
|---|---|---|---|
| dense | 0.397 [0.283, 0.513] | **0.415 [0.309, 0.531]** | 0.380 [0.272, 0.505] |
| bm25 | 0.284 [0.180, 0.405] | 0.359 [0.250, 0.479] | **0.376 [0.258, 0.510]** |
| hybrid | 0.402 [0.292, 0.521] | 0.369 [0.265, 0.479] | **0.409 [0.296, 0.538]** |

### ndcg@10

| mode | B0 | B1 | E1 |
|---|---|---|---|
| dense | 0.411 [0.300, 0.511] | **0.413 [0.315, 0.509]** | 0.366 [0.276, 0.461] |
| bm25 | 0.301 [0.209, 0.398] | **0.372 [0.271, 0.468]** | 0.367 [0.265, 0.469] |
| hybrid | **0.420 [0.320, 0.524]** | 0.397 [0.298, 0.499] | 0.412 [0.309, 0.514] |

## Per QASPER answer type — hybrid

### recall@10

| answer type (n) | B0 | B1 | E1 |
|---|---|---|---|
| extractive (38) | **0.588 [0.453, 0.720]** | 0.557 [0.424, 0.684] | 0.585 [0.447, 0.712] |
| abstractive (1) | **1.000 [1.000, 1.000]** | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| boolean (5) | **0.600 [0.200, 1.000]** | 0.600 [0.200, 1.000] | 0.600 [0.200, 1.000] |

### mrr@10

| answer type (n) | B0 | B1 | E1 |
|---|---|---|---|
| extractive (38) | 0.414 [0.290, 0.540] | 0.398 [0.274, 0.522] | **0.448 [0.309, 0.580]** |
| abstractive (1) | **0.250 [0.250, 0.250]** | 0.250 [0.250, 0.250] | 0.111 [0.111, 0.111] |
| boolean (5) | **0.333 [0.033, 0.700]** | 0.175 [0.025, 0.325] | 0.167 [0.033, 0.300] |

## Written analysis — baselines (B0 vs B1)

- **Naive enrichment (B1) does not beat plain chunks (B0)** on this text-only subset: B1 is higher on 1/5 metrics, lower on 4. It is not significantly different on any metric (wide CIs). With only 44 queries the 95% CIs are very wide and mostly overlap, so read directionally.
- **Dense side:** B1 dense Recall@10 0.579 vs B0 0.593; BM25 barely moves because appending questions to the chunk document adds little over the chunk text. The questions-only index trades top-rank precision for coverage (consistent with the earlier 15-paper pilot).
- **Why, not a bug:** QASPER queries are written from title+abstract only, so they are lexically distant from passage-specific naive questions. Index integrity verified (chunk→self rank 1; generated-question→parent rank 1; 0 gold ids missing from the corpus).

## Written analysis — Experiment 1 (semantic question type)

- **Did type-stratification (E1) help?** On this text-only subset (10 papers, 44 queries) the arms are statistically indistinguishable: E1 is higher than B1 on 4/5 metrics (lower on 1), and vs B0 E1 is higher on 2/5 (lower on 3); no metric reaches significance vs B1 or B0. Directionally E1 ≈ B0 and edges B1 on the ranking metrics (MRR@10, nDCG@10). Read as directional — CIs overlap at n=44.
- **What E1 does mechanically:** it forces balanced type coverage. Naive B1 over-produces CONCEPT (31% of its questions) and under-produces DISJUNCTIVE/PROCEDURAL/COMPARISON; E1 spreads questions across all 9 types. Same-type-on-gold **coverage rises 91% (B1) → 98% (E1)**.
- **Cross-tab (directionally present > absent, still underpowered):** splitting B1 queries by whether a same-type question sits on a gold chunk gives present 0.7 (n=40) vs absent 0.5 (n=4) — with only n=4 'absent' cases this is suggestive, not conclusive. Naive generation already covers most query types on the gold chunk, which is why the 'absent' cell stays small.
- **Where the gain would come from / what to re-run:** the type effect on this text-answerable sample is neutral-to-slightly-positive on ranking; to detect it you need more 'absent' cases (scale papers, or oversample rarer query types). The same 10-paper corpus now powers Experiments 2–5, which target the vocabulary/scope/style gaps more directly than type alone.

## Experiment 1 — semantic-type cross-tab

Retrieval success (**hit@10**, on the naive B1 arm) by real-query type × whether a same-type generated question sat on a gold chunk. Coverage: B1 91% → E1 98%.

| query type | n | same-type present (n, hit) | same-type absent (n, hit) |
|---|---|---|---|
| VERIFICATION | 6 | 4, 0.5 | 2, 0.5 |
| CONCEPT | 12 | 11, 0.727 | 1, 1.0 |
| EXTENT | 6 | 6, 0.667 | 0, — |
| EXAMPLE | 12 | 12, 0.583 | 0, — |
| COMPARISON | 2 | 1, 1.0 | 1, 0.0 |
| CAUSE | 1 | 1, 1.0 | 0, — |
| CONSEQUENCE | 2 | 2, 1.0 | 0, — |
| PROCEDURAL | 3 | 3, 1.0 | 0, — |