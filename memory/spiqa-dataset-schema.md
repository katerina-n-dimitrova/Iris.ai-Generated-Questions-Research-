---
name: spiqa-dataset-schema
description: The three small SPIQA splits (test-A/B/C, val) use different JSON schemas; only test-B and test-C carry paper body text.
metadata:
  type: reference
---

SPIQA on HF is `google/spiqa`. QA metadata is in per-split JSON files
(test-A/SPIQA_testA.json, test-B/SPIQA_testB.json, test-C/SPIQA_testC.json,
train_val/SPIQA_val.json + SPIQA_train.json). Download only the split JSONs via
`huggingface_hub.hf_hub_download` — never the 208 MB train JSON or the multi-GB
image zips.

Three different schemas:
- **test-B** (65 papers): flat — `passages[]` (body paragraphs), `all_figures_tables{fname:caption}`,
  `figure_types{}`, and PARALLEL per-question lists: `question[]`, `composition[]` (answer),
  `evidential_info[]` (list of {context} verbatim excerpts), `referred_figures_tables[]`
  (gold fig/table ids per question), `Is_figure/table_in_evidence[]`.
- **test-C** (314 papers): `paper_title`, `abstract`, `full_text[]`={section_name,paragraphs[]},
  `figures_and_tables[]`={file,caption}, `question[]`, `answer[]`={free_form_answer,evidence},
  `referred_figures_tables[]`.
- **test-A / val** (118 / 200 papers): image-QA — `all_figures{fname:{caption,content_type,figure_type}}`
  and `qa[]`={question,answer,explanation}. NO body text and NO per-question gold figure ids.

Gold for retrieval eval (test-B/C): referred_figures_tables = exact figure/table gold;
evidential_info excerpts are verbatim from passages so match text chunks by normalized substring.
Fallback = paper-level relevance when a question has no locatable chunk-level gold.
See [[spiqa-num-questions-experiment]].
