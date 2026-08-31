---
name: multihoprag-baseline-harness
description: "src/multihoprag/ study of WHICH question type to generate on MultiHop-RAG (news, multi-evidence); B0/B1 harness validated on a 10-article pilot (69 queries); Exp 1-5 pending."
metadata: 
  node_type: memory
  type: project
  originSessionId: 165b2748-63ab-4c79-906a-9066e7b90d23
---

New sub-project `src/multihoprag/` — same research question as [[qasper-baseline-harness]]
(**what KIND of question to generate** for doc2query enrichment) on **MultiHop-RAG**
(Tang & Yang 2024, HF `yixuantt/MultiHopRAG`: 609 news articles + 2,556 queries).
Built as a close adaptation of the qasper harness. Hybrid **dense + BM25 + RRF (k=60)**;
modular: `mhrag_config` (fixed factors + Arm registry + `validate_arm`), `mhrag_data`,
`mhrag_generate` (prompt registry, cached/resumable, retry-once + failure log),
`mhrag_index`, `mhrag_retrieval` (saves rankings), `mhrag_eval` (bootstrap CIs +
paired significance), `mhrag_report` (results.md + self-contained results.html,
rebuildable), `run_mhrag`.

**Fixed-factor invariant** (same as qasper): enrichment arms embed **ONLY
generated-question vectors** (chunk dense score = max over its questions); chunk
text embedded only in **B0** and the Exp-4 variants **E4b/E4e/E4f** (`CHUNK_EMBED_OK`).

**Dataset — query-first selection** (evidence spans 2-4 articles, so random
articles leave ~no answerable query): drop null queries → shuffle (seed 42) →
greedily accept a query iff its evidence articles FIT within `ARTICLE_BUDGET`
(fit-only, gives exactly the budget) → then keep every remaining query fully
covered. Scale via ONE knob `MHRAG_ARTICLE_BUDGET` (pilot=10; 20→124q, 50→529q).
Article id = url. **Pilot (budget 10): 10 articles → 180 chunks (~18/article, ~89
words), 69 queries (inference 29 / temporal 23 / comparison 17), avg 2.71 gold
chunks/query.** Gold = chunks containing an evidence `fact` snippet (whitespace-norm
substring, else fuzzy>=0.9); **all 187 facts matched, 0 unmatched, 0 multi-chunk**.
Chunking: split on blank lines, merge paragraphs <40 tokens with a neighbour.

**Metrics differ from qasper** — queries are multi-evidence, so PRIMARY = **Evidence
Recall@k** (fraction of gold chunks in top k; k=2,5,10), plus **Hit@k**, MRR@10,
nDCG@10. Reported overall / per query type (inference/comparison/temporal) / per
mode (dense/bm25/hybrid). 1000-resample bootstrap CIs, paired-bootstrap significance.

**Baseline result (validated, hybrid mean [95% CI], n=69):** naive enrichment B1
does NOT beat plain chunks B0 — directionally B1 slightly BEHIND on early precision
(ER@2 0.374 vs 0.406, MRR@10 0.716 vs 0.768, hit@2 0.812 vs 0.899) but **NO metric
significant** (ER@5 Δ=-0.021, p=0.47, CI includes 0). Hit@10=1.0 both (tiny corpus).
**Mode breakdown localises it:** B1 questions-only DENSE is worse than B0 chunk
vectors (dense MRR 0.586 vs 0.699) while appending questions to BM25 HELPS sparse
(bm25 ER@5 0.610 vs 0.556); hybrid nets ~even. Reproduces the qasper/peerqa
"questions-only loses early precision, BM25-append helps" pattern. Integrity passes
(chunk→self rank1; gen-q→parent rank1; 0 gold missing). B1 gen: 1800 q, 0% fail, $0.02.

**Why:** internship research programme (Iris AI). **How to apply:** `cd src/multihoprag
&& python run_mhrag.py` (cached; `--eval-only` rebuilds from saved rankings). gpt-4o-mini
@0.3, Octen-0.6B. **User gate:** validate B0/B1 before building the 5 experiments.
**Next:** the 5 experiments — semantic type (Cao&Wang ontology + cross-tab), scope
(local/summary), explicitness (implicit vocab-bridging), surface-form/placement
(Exp-4 a-f incl. chunk-embed variants), style-match (few-shot on real out-of-eval
MultiHop-RAG queries, assert no leakage). Exp 3 & 5 target the dense deficit. Scale
to 50-100 articles to tighten CIs. See [[qasper-baseline-harness]],
[[peerqa-questions-only-experiment]].
