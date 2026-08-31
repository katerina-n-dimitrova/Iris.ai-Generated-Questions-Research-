---
name: docbank-enrichment-experiment
description: DocBank generated-question enrichment; questions-only BEATS chunk-text baseline at all counts (opposite of PeerQA) — but synthetic-eval/enrichment share an LLM origin, which flatters it.
metadata:
  type: project
---

Sub-project `src/docbank/` (docbank_config, docbank_loader, docbank_chunker,
docbank_qa, docbank_experiment, run_docbank, docbank_html). Reuses PeerQA
doc2query gen + metric helpers ([[peerqa-questions-only-experiment]]). Report:
`report/docbank_generated_questions_results.html` (13 sections + 4 inline-SVG
distribution charts). Results: `results/docbank/docbank_results.json` + eval set
`docbank_15docs_eval_qa.{json,csv}`.

**Dataset access (see also [[docbank-dataset-access]]):** DocBank = arXiv/LaTeX
pages, token-level layout labels (title/section/paragraph/table/equation/caption/
reference/…). Parquet mirror `maveriq/DocBank` is token-level + ANONYMISED (no
doc/page id) → can't group into documents. Original `liminghao1630/DocBank`
`DocBank_500K_txt.zip` (3.17GB) filenames encode arXiv id + page, so we used
`remotezip` to fetch only the zip central directory + the 15 selected docs' small
.txt files (a few MB, never the full zip). No image analysis needed — token+label
stream carries all text/structure.

**Setup:** 15 docs (6–8 pages each, early 1401.xxxx math papers → equation-heavy,
table-sparse: 3 tables), 104 pages, 348 layout-aware chunks (paragraph 208,
equation 64, mixed 45, caption 28, table 3; ~500 tok flow chunks + table/caption/
big-equation kept as own units w/ heading+context). Synthetic QA eval (LLM,
grounded, one gold chunk each, kept SEPARATE from enrichment questions): 118 QA.
4729 enrichment questions ($0.076). Octen-0.6B, gpt-4o-mini.

**Key result — questions-only WINS here (opposite of PeerQA):**
baseline (chunk-text) nDCG@10 0.646 / Hit@1 0.483. q5 0.707, q10 0.713, **q13 0.720
(best) / Hit@1 0.593**, q15 0.718 — monotonic to q13 then flat; fused 0.693. All
q-conditions beat baseline on EVERY metric incl. Hit@1.

**Critical caveat:** eval questions AND enrichment questions are both LLM-generated
from the same chunks → generator overlap flatters questions-only retrieval. This is
almost certainly why DocBank (synthetic QA) favors questions-only while
[[peerqa-questions-only-experiment]] (real reviewer questions) did not. Treat
absolute numbers as optimistic; relative q5→q15 trend is the real signal.

**Recommendation given:** prefer fused over pure questions-only; use real/citation
eval labels to remove generator bias; sample table/figure-rich docs (this sample
under-tests tables); exploit layout type (type-specific questions/routing).

**Chunk-size follow-up** (`docbank_chunksize_experiment.py`, `run_docbank_chunksize.py`,
`docbank_chunksize_html.py`; report `report/docbank_chunksize_results.html`;
results `docbank_chunksize_results.json`; question cache
`chunksize_enrichment_questions.jsonl`). Replicated the PeerQA chunk-size study on
DocBank: 5 chunkings (fixed 200/400/600/800 + section-aware variable) × q5/q10/q13/q15
+ adaptive. Reused the 118 QA as queries, **remapping each gold onto every chunking
by evidence-text substring match** (88 map in all 5 = common set). Key findings:
- **Best count RISES with chunk size** (200→q13, 400→q13, 600→q15, 800→q15) — bigger
  chunks want more questions. Best overall = **chunk_size_600_q15 nDCG@10 0.710**.
- Baselines by size: 200 0.607, 400 0.636, 600 0.616, 800 0.593 (size moves nDCG
  ~0.042; count within a size up to ~0.116 — count is the bigger lever here).
- **Adaptive length-based ("bigger⇒more", avg 5.98 q/chunk) beats fixed q10** (0.664
  vs 0.639) at FEWER embeddings (1975 vs 3176). Adaptive density 0.678. **Fused 0.694**
  (cheapest strong). Unlike PeerQA, questions-only beats chunk-text here — same
  synthetic-eval/generator-overlap caveat applies. See [[peerqa-chunksize-experiment]].
