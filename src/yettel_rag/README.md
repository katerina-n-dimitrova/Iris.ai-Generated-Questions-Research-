# Yettel Bulgaria MultiHop-RAG-style dataset

This pipeline creates a Bulgarian, prose-only corpus from the public Yettel
Bulgaria corporate sitemap. It keeps one HTML page per document and applies a
strict 1,500–5,000 `cl100k_base` token filter. It does not pad or concatenate
short pages.

## Build

```bash
cd context-enrichment-rag
.venv/bin/python src/yettel_rag/build_corpus.py
.venv/bin/python src/yettel_rag/generate_questions.py
.venv/bin/python src/yettel_rag/validate_dataset.py
```

Use `--refresh` to redownload the public pages. Cached HTML is stored under
`data/raw/yettel_bg/`; generated artifacts are under
`data/processed/yettel_bg/`.

Outputs:

- `corpus.json`: array compatible with the MultiHop-RAG corpus layout, with an
  additional stable `document_id` and explicit `date`.
- `documents.jsonl`: full metadata, clean body, token count, and content hash.
- `chunks_1024.jsonl`: 1,024-token chunks with 128-token overlap.
- `crawl_audit.jsonl`: inclusion/exclusion decision for every sitemap URL.
- `manifest.json`: provenance, counts, extraction rules, and chunking settings.
- `questions.jsonl` and `MultiHopRAG.json`: evaluation-only questions with
  answers, 2–4-document gold evidence, and canonical chunk IDs.
- `question_manifest.json`: generation provenance and exact type/hop counts.

Question generation is resumable through
`question_generation_checkpoint.jsonl`. Its fixed seed and evaluation-only
prompt are deliberately separate from any questions used for chunk enrichment.

Yettel usually omits a publication date from article HTML. The `date` field
therefore contains the sitemap's `lastmod` value and `date_type` is set to
`last_modified`; it must not be interpreted as a publication timestamp.

Tables, figures, captions, images, audio/video, forms, navigation, headers,
footers, scripts, and other non-prose elements are excluded by the parser.
