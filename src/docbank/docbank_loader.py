"""
DocBank loader — samples 15 full documents (grouped by arXiv doc id) and parses
their per-page token/layout .txt files into ordered layout blocks.

Only a few MB are downloaded: the zip central directory (namelist) + the small
.txt files of the selected documents, all via HTTP range requests. Everything is
cached to disk so re-runs are offline.

Each .txt line: token \t x0 \t y0 \t x1 \t y1 \t R \t G \t B \t font \t label
We drop graphical placeholder tokens (##LT...##) and merge consecutive same-label
tokens on a page into a "block" (title / section / paragraph / table / equation /
caption / reference / figure ...), preserving reading order and page number.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import docbank_config as C

_DOCID_RE = re.compile(r"_(\d+)\.txt$")


def _docid(name: str) -> str:
    """Strip the trailing _<page>.txt to get the document id."""
    return _DOCID_RE.sub("", name.split("/")[-1])


def _page_num(name: str) -> int:
    m = _DOCID_RE.search(name)
    return int(m.group(1)) if m else 0


def _arxiv_id(docid: str) -> str:
    m = re.search(r"(\d{4}\.\d{4,5})", docid)
    return m.group(1) if m else docid


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    label: str  # raw DocBank label
    ltype: str  # coarse layout type (LAYOUT_MAP)
    text: str
    page: int


@dataclass
class Document:
    doc_id: str
    arxiv_id: str
    pages: List[int]
    blocks: List[Block] = field(default_factory=list)

    def to_row(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "arxiv_id": self.arxiv_id,
            "pages": self.pages,
            "blocks": [asdict(b) for b in self.blocks],
        }


# --------------------------------------------------------------------------- #
# Remote namelist (cached)
# --------------------------------------------------------------------------- #
def _load_names() -> List[str]:
    if C.NAMES_CACHE.exists():
        return [l.strip() for l in C.NAMES_CACHE.open(encoding="utf-8") if l.strip()]
    from remotezip import RemoteZip

    print("[docbank] fetching zip central directory (namelist)...")
    with RemoteZip(C.TXT_ZIP_URL) as z:
        names = [n for n in z.namelist() if n.endswith(".txt")]
    C.NAMES_CACHE.write_text("\n".join(names), encoding="utf-8")
    return names


def select_documents(num_docs: int = C.NUM_DOCS) -> Dict[str, List[str]]:
    """Return {doc_id: [txt member names sorted by page]} for the chosen docs."""
    names = _load_names()
    by_doc: Dict[str, List[str]] = {}
    for n in names:
        by_doc.setdefault(_docid(n), []).append(n)
    eligible = sorted(
        (d for d, ns in by_doc.items() if C.MIN_PAGES <= len(ns) <= C.MAX_PAGES)
    )
    chosen = eligible[:num_docs]
    return {d: sorted(by_doc[d], key=_page_num) for d in chosen}


# --------------------------------------------------------------------------- #
# Extract + parse
# --------------------------------------------------------------------------- #
def _extract(members: List[str]) -> None:
    """Download any not-yet-cached member .txt files via remotezip range reads."""
    missing = [m for m in members if not (C.TXT_DIR / m.split("/")[-1]).exists()]
    if not missing:
        return
    from remotezip import RemoteZip

    print(f"[docbank] extracting {len(missing)} page files via range requests...")
    with RemoteZip(C.TXT_ZIP_URL) as z:
        for m in missing:
            data = z.read(m)
            (C.TXT_DIR / m.split("/")[-1]).write_bytes(data)


def _parse_page(path: Path, page: int) -> List[Block]:
    blocks: List[Block] = []
    cur_label = None
    cur_tokens: List[str] = []

    def flush():
        nonlocal cur_tokens, cur_label
        if cur_tokens and cur_label:
            text = " ".join(cur_tokens).strip()
            # tidy spacing before punctuation
            text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
            if text:
                blocks.append(
                    Block(
                        label=cur_label,
                        ltype=C.LAYOUT_MAP.get(cur_label, "other"),
                        text=text,
                        page=page,
                    )
                )
        cur_tokens = []

    for line in path.open(encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 10:
            continue
        token, label = parts[0], parts[9]
        if token in C.PLACEHOLDER_TOKENS or not token.strip():
            continue
        if label != cur_label:
            flush()
            cur_label = label
        cur_tokens.append(token)
    flush()
    return blocks


def load_documents(num_docs: int = C.NUM_DOCS) -> List[Document]:
    sel = select_documents(num_docs)
    all_members = [m for ms in sel.values() for m in ms]
    _extract(all_members)

    docs: List[Document] = []
    for doc_id, members in sel.items():
        blocks: List[Block] = []
        pages: List[int] = []
        for m in members:
            page = _page_num(m)
            pages.append(page)
            blocks.extend(_parse_page(C.TXT_DIR / m.split("/")[-1], page))
        docs.append(
            Document(
                doc_id=doc_id,
                arxiv_id=_arxiv_id(doc_id),
                pages=sorted(pages),
                blocks=blocks,
            )
        )
    return docs


def save_documents(docs: List[Document]) -> Path:
    out = C.PROCESSED_DIR / "documents.json"
    json.dump(
        [d.to_row() for d in docs],
        out.open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    return out


def dataset_summary(docs: List[Document]) -> Dict:
    import collections

    label_counts = collections.Counter()
    ltype_counts = collections.Counter()
    for d in docs:
        for b in d.blocks:
            label_counts[b.label] += 1
            ltype_counts[b.ltype] += 1
    return {
        "num_documents": len(docs),
        "num_pages_total": sum(len(d.pages) for d in docs),
        "avg_pages_per_doc": round(
            sum(len(d.pages) for d in docs) / max(len(docs), 1), 1
        ),
        "num_blocks_total": sum(len(d.blocks) for d in docs),
        "block_label_counts": dict(label_counts.most_common()),
        "layout_type_counts": dict(ltype_counts.most_common()),
        "documents": [
            {
                "doc_id": d.doc_id,
                "arxiv_id": d.arxiv_id,
                "num_pages": len(d.pages),
                "num_blocks": len(d.blocks),
            }
            for d in docs
        ],
    }


if __name__ == "__main__":
    docs = load_documents()
    save_documents(docs)
    summ = dataset_summary(docs)
    print(json.dumps(summ, indent=2)[:2500])
