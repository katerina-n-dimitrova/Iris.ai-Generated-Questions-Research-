---
name: peerqa-questions-only-experiment
description: PeerQA doc2query where the index holds ONLY generated questions (no chunk text); wins recall, loses early precision, monotonic in count.
metadata:
  type: project
---

New sub-project `src/peerqa/` (peerqa_config, peerqa_data, peerqa_experiment,
run_peerqa, peerqa_html). Mirrors the SPIQA pipeline but the key twist: the
index embeds **ONLY the generated questions, never the chunk text** (pure
doc2query). A chunk-text dense index is kept as `baseline` only, so "does
questions-only beat the text" is answerable. Conditions: baseline(0)/q5/q10/q13;
13-question pool generated once per chunk, q5/q10 are first-k subsets. Report:
`report/peerqa_generated_questions_results.html` (7 sections, data-driven from
`results/peerqa/peerqa_results.json`).

**Setup:** 15 text-available papers (top by usable-query count), 333 chunks
(~500 tok, cap 600, 100 overlap — packed from PeerQA's sentence units), 65 eval
queries, 4276 questions ($0.08, ~2min, gpt-4o-mini), Octen-0.6B embedder.

**Key result (baseline nDCG@10 0.530 / Hit@1 0.323 / Hit@5 0.615 / Hit@10 0.754):**
- **Questions-only does NOT beat chunk text on early precision** — q13 Hit@1
  0.292, MRR 0.435, nDCG@10 0.519 all below baseline.
- **But it WINS on deeper recall** — q13 Hit@5 0.692 (+0.077), Hit@10 0.800
  (+0.046), recall@10 0.717 (+0.023). Many small question vectors cast a wider net.
- **Cleanly MONOTONIC in count** (q5<q10<q13 on every metric) — unlike single-vector
  append which dilutes, each question is its own vector so more only adds coverage;
  no saturation dip within 5→13. q5 badly under-covers (nDCG 0.394).
- Cost: q5→q13 = 5.0×–12.84× baseline embeddings but only 1.5×–2.8× index size
  (question vectors store no chunk text). Search p95 ~6–8ms.

**Why:** internship research question — isolate embedding-only-questions. **How to
apply:** `python run_peerqa.py --num-papers 15` from `src/peerqa/` (cached/resumable).
**Next:** recommended hybrid (keep text vector + fuse question scores, per
[[spiqa-statement-q10-findings]]), scale to 90/208 papers, BM25 expansion,
paragraph-level qrels. See [[peerqa-dataset-schema]], [[spiqa-num-questions-experiment]].
