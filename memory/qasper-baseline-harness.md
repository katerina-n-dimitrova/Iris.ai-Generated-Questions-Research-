---
name: qasper-baseline-harness
description: src/qasper/ study of WHICH question type to generate; harness validated on a 15-paper pilot, but user pinned the real run to 4 STRICTLY TEXT-ONLY QASPER papers (n=18 queries) — too small to separate arms. B0/B1/E1 built; Exp 2-5 pending.
metadata:
  type: project
---

**CURRENT corpus = 10 `text_answerable` dev papers (the run for Exp 1-5).** The
user wants the study on TEXT only. Strict "no floats in the PDF" (`text_only`) gives
only ~4 QASPER papers (too few, n=18, all CIs overlap) — kept as a mode but not used.
Default is now `SELECTION_MODE=text_answerable`: corpus = text paragraphs only AND
every EVALUATED question is fully text-answerable (float-dependent questions dropped
via `REQUIRE_FULLY_TEXT_QUERIES`), on formula-light papers (fraction<=0.10); seed-42
sample of NUM_PAPERS=10 from the ~59 qualifying dev papers. → **321 chunks, 44
queries** (extractive 38, boolean 5, abstractive 1 — extractive-heavy, so per-type
breakdown is thin outside extractive). Papers: 1603.01514, 1612.03226, 1612.08205,
1701.08229, 1707.05236, 1707.06806, 1712.00991, 1906.07662, 1910.03634, 2002.04181.
Umbrella runner: `python run_qasper.py` (B0,B1,E1 + Exp-1 + report). Modes/thresholds
in qasper_config (QASPER_SELECTION, QASPER_TA_MAX_FORMULA, QASPER_TA_MIN_QUERIES).

**Exp 1 (semantic type) built (E1 arm, `typed` prompt):** 9 Cao&Wang types (JUDGMENTAL
skipped), 10 slots allocated 1/type + extras to the most common real-query types,
expanded into explicit numbered slots so the model returns exactly 10 (an earlier
"2x CONCEPT" allocation made gpt-4o-mini emit only 9 → 34% shortfall; the enumerated
template fixed it to 0%). LLM classifier (temp 0, cached) labels queries + generated
questions → the cross-tab. On the 10-paper text_answerable set (44 q) B0≈B1≈E1 — NO
metric is significant (paired bootstrap); directionally E1≈B0 and edges B1 on MRR@10 /
nDCG@10. Coverage of same-type-on-gold rises B1 91% → E1 98%; naive B1 over-produces
CONCEPT (~31%). Cross-tab present 0.70 (n=40) > absent 0.50 (n=4) — directionally
supportive but underpowered (only 4 'absent'). Net: at this scale question TYPE is
neutral-to-slightly-positive; Exp 2-5 (scope/explicitness/surface/style) target the
vocab & distribution gaps more directly. Earlier 15-paper mixed pilot had E1 > B1.

New sub-project `src/qasper/` isolating **what KIND of question to generate** for
doc2query enrichment (vs prior sub-projects that varied count/scope). Hybrid
retriever: **dense + BM25 + RRF (k=60)**. Modular by design: `qasper_config`
(fixed factors + Arm registry + `validate_arm` invariant), `qasper_data`,
`qasper_generate` (prompt registry, cached/resumable, retry-once + failure log),
`qasper_index`, `qasper_retrieval` (saves rankings), `qasper_eval` (bootstrap
CIs + paired significance), `qasper_report` (results.md + self-contained
results.html, rebuildable from saved metrics), `run_qasper_baselines`.

**Fixed-factor invariant (locked by user mid-build):** enrichment arms embed
**ONLY generated-question vectors** (chunk dense score = max over its questions);
chunk text is embedded in **exactly two** arms — B0 and a future Exp‑4 variant
`E4f` (questions+chunk). `validate_arm` raises if any other arm sets
`embeds_chunk`. Chunk text otherwise used only for BM25 (+appended questions) and
gold matching.

**Data:** canonical AllenAI QASPER v0.3 dev JSON (HF loader-script is dead on
`datasets` 5.0 — downloaded the S3 tgz instead). Seed 42 → **15 papers, 790 text
chunks** (paragraph=chunk + abstract; float/table + <6-word + <0.5-alpha dropped;
0 gold lost to filtering), **50 queries** (extractive 30 / abstractive 12 /
boolean 8; unanswerable + figure-only skipped), gold = union of evidence
paragraphs across annotators (142 matched, 3 unmatched ~98%). Scale via one knob:
`QASPER_NUM_PAPERS=0`. Note: 790 chunks < the 1.5–2.5k estimate (these dev papers
are short).

**Baseline result (validated harness, hybrid mean [95% CI]):** B1 (naive 10-q
enrichment) is **significantly WORSE than B0** (plain chunks) on early precision —
Recall@1 0.124→0.013, Recall@5 0.285→0.141, MRR@10 0.279→0.151, nDCG@10
0.250→0.166 (paired bootstrap CIs exclude 0); Recall@10 ~tied (0.361 vs 0.340).
Loss is on the DENSE side (BM25 barely moves). Deep recall flips: B1 Recall@100
0.686 > B0 0.643 — **exactly the [[peerqa-questions-only-experiment]] pattern**
(questions-only: worse top-rank precision, wider net). Integrity checks pass
(chunk→self rank1; gen-q→parent rank1; 0/106 gold missing). Absolute numbers low
because QASPER queries are written from title+abstract only → lexically distant
from passage-specific naive questions (this is the whole motivation).

**Why:** internship research programme. **How to apply:** `cd src/qasper &&
python run_qasper_baselines.py` (cached; `--eval-only` rebuilds metrics from saved
rankings in ~4s). gpt-4o-mini@0.3, Octen-0.6B, $0.09 gen. **Next:** the 5
experiments (semantic type / scope / explicitness / surface-form / style-match) —
Exp 3 (implicit, vocab-bridging) and Exp 5 (style-match to QASPER distribution)
are the likely fixes for B1's precision deficit; scale to full dev split to
tighten CIs (n=50, boolean n=8 → wide). See [[peerqa-chunksize-experiment]],
[[docbank-enrichment-experiment]].
