---
name: docbank-dataset-access
description: How to sample DocBank without downloading 500K pages — remotezip range-reads on the original txt zip (filenames carry arXiv doc id + page).
metadata:
  type: reference
---

DocBank (Li et al., 2020; arXiv:2006.01038) = 500K arXiv/LaTeX pages, token-level
layout annotations. Two HF sources:
- **`maveriq/DocBank`** — parquet, streamable, but TOKEN-LEVEL and ANONYMISED:
  fields `token, bounding_box, color, font, label, image`; no document/page id, so
  pages can't be grouped into documents. Group tokens→page only by image identity.
- **`liminghao1630/DocBank`** — original. `DocBank_500K_txt.zip` (3.17 GB) has one
  .txt per page; **filenames encode arXiv id + page**, e.g.
  `1.tar_1401.0091.gz_..._arxiv_7.txt` (doc id = strip trailing `_<page>.txt`;
  arXiv id = the `\d{4}\.\d{4,5}`). Also `..._ori_img.zip.001..010` (images, skip).

**Sampling technique (compliant with "don't download the full dataset"):** the HF
LFS CDN supports HTTP byte ranges (`accept-ranges: bytes`). Use `remotezip`
(`pip install remotezip`) on the resolve URL
`https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_txt.zip`:
`z.namelist()` downloads only the zip central directory (~6s, 500001 entries),
then `z.read(member)` range-fetches individual small .txt files. Total a few MB,
never the 3.17 GB. Cache namelist + extracted files to disk.

txt line format (tab-separated):
`token \t x0 \t y0 \t x1 \t y1 \t R \t G \t B \t font \t label`. Drop graphical
placeholders `##LTLine##`/`##LTFigure##`/etc. Merge consecutive same-label tokens
into layout blocks. Pages-per-doc: 42K docs have 1 page, ~19K have 6–10 pages
(good for "full document" sampling). Used by [[docbank-enrichment-experiment]].
Same remotezip trick works for any HF-hosted zip with range support.
