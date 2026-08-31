---
name: peerqa-dataset-schema
description: PeerQA ships paper text pre-segmented at SENTENCE level; gold evidence is sentence-idx sets; only 90 papers have redistributable full text.
metadata:
  type: reference
---

PeerQA (Baumgärtner et al., 2025; arXiv:2502.13668), HF `UKPLab/PeerQA`. The HF
repo is a loading SCRIPT (`PeerQA.py`) — dead on `datasets>=3` (5.0.0 here drops
script support). Download the data zip directly instead:
`https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4467/peerqa-data-v1.0.zip?sequence=5&isAllowed=y`
→ gives `qa.jsonl`, `papers.jsonl`, `qa-augmented-answers.jsonl` (NO pre-built
qrels; the script derives them). Code lives in `src/peerqa/peerqa_data.py`.

- **papers.jsonl** (24,265 rows): one row per SENTENCE — `paper_id`, `idx`
  (global sentence index in paper), `pidx` (paragraph), `sidx` (sentence-in-para),
  `type` (sentence/heading/table/figure/formula/list_item/title), `content`,
  `last_heading`. So docs are ALREADY chunked, at sentence granularity, NO overlap.
- **qa.jsonl** (579 questions): `paper_id`, `question_id`, `question`,
  `answer_free_form`, `answer_evidence_mapped` = list of `{sentence, idx:[...]}`
  where `idx` are the GOLD evidence sentence idxs (relevance labels; verified
  389/389 in-range). `answerable` / `answerable_mapped` flags.

Sources by prefix: `openreview/` (369 q, ICLR/NeurIPS), `nlpeer/` (177 q,
ARR-22/COLING2020/F1000-22/PeerRead), `egu/` (33 q). **Catch:** `papers.jsonl`
holds full text for only the 90 nlpeer+egu papers — the 118 openreview papers
need the `papers-all` config that FETCHES full text from arXiv/OpenReview at load
time ("PeerQA from arXiv"). Usable retrieval queries = answerable_mapped + mapped
evidence + text present = **136 queries over 70 papers** (of the nominal 208).
See [[peerqa-questions-only-experiment]], schema cousin [[spiqa-dataset-schema]].
