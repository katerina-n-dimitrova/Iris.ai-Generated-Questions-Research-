#!/usr/bin/env python3
"""Build the strict text-only Yettel Bulgaria corpus and 1,024-token chunks.

The source inventory is Yettel's own corporate sitemap.  Each output document
maps to one HTML URL; short pages are never concatenated or padded.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import tiktoken

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "yettel_bg"
OUT_DIR = ROOT / "data" / "processed" / "yettel_bg"
SITEMAP_URL = "https://www.yettel.bg/bg/sitemap-corporate.xml"
MIN_TOKENS = 1_500
MAX_TOKENS = 5_000
CHUNK_TOKENS = 1_024
CHUNK_OVERLAP = 128
TOKENIZER = "cl100k_base"
TARGET_DOCUMENTS = 340
USER_AGENT = "Yettel-RAG-Research/1.0 (public HTML corpus; contact: dataset-builder)"

_WS = re.compile(r"[\t\f\v ]+")
_BLANKS = re.compile(r"\n\s*\n+")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_sitemap(payload: bytes) -> list[dict]:
    root = ET.fromstring(payload)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    rows = []
    for item in root.findall("sm:url", namespace):
        location = item.findtext("sm:loc", default="", namespaces=namespace).strip()
        modified = item.findtext("sm:lastmod", default="", namespaces=namespace).strip()
        if location:
            rows.append({"url": location, "last_modified": modified})
    return rows


class ArticleExtractor(HTMLParser):
    """Extract visible prose from the first article inside the page main area."""

    SKIP_TAGS = {
        "script",
        "style",
        "svg",
        "img",
        "picture",
        "source",
        "video",
        "audio",
        "iframe",
        "canvas",
        "table",
        "figure",
        "figcaption",
        "form",
        "button",
        "nav",
        "footer",
        "noscript",
        "header",
    }
    BLOCK_TAGS = {
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "li",
        "br",
        "dd",
        "dt",
        "blockquote",
    }
    SKIP_CLASSES = {
        "breadcrumb",
        "breadcrumbs",
        "share-buttons",
        "social-share",
        "news-gallery-main",
        "news-gallery-thumbnails-wrapper",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.main_depth = 0
        self.article_depth = 0
        self.skip_depth = 0
        self.skip_stack: list[bool] = []
        self.done = False
        self.parts: list[str] = []
        self.title = ""
        self.page_language = ""
        self._heading_depth = 0
        self._heading_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "html":
            self.page_language = values.get("lang") or ""
        if tag == "main" and not self.done and not self.main_depth:
            self.main_depth = 1
        elif self.main_depth and not self.article_depth and tag == "article":
            self.article_depth = 1
        elif self.article_depth and tag == "article":
            self.article_depth += 1

        if self.article_depth and tag != "article":
            skipped = bool(
                self.skip_depth or tag in self.SKIP_TAGS or classes & self.SKIP_CLASSES
            )
            self.skip_stack.append(skipped)
            if skipped:
                self.skip_depth += 1
        if self.article_depth and not self.skip_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if (
            self.article_depth
            and not self.skip_depth
            and tag == "h1"
            and not self.title
        ):
            self._heading_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self._heading_depth and tag == "h1":
            self.title = " ".join("".join(self._heading_parts).split())
            self._heading_depth = 0
        if self.article_depth and tag != "article" and self.skip_stack:
            skipped = self.skip_stack.pop()
            if skipped:
                self.skip_depth -= 1
        if self.article_depth and tag == "article":
            self.article_depth -= 1
            if not self.article_depth:
                self.done = True
        if self.main_depth and tag == "main":
            self.main_depth = 0

    def handle_data(self, data: str) -> None:
        if self.article_depth and not self.skip_depth:
            self.parts.append(data)
            if self._heading_depth:
                self._heading_parts.append(data)


def clean_text(parts: list[str]) -> str:
    raw = html.unescape("".join(parts)).replace("\r", "\n").replace("\u00a0", " ")
    lines = [_WS.sub(" ", line).strip() for line in raw.splitlines()]
    text = "\n".join(lines)
    text = _BLANKS.sub("\n\n", text).strip()
    # Remove Drupal's occasional anonymous-user metadata if a template placed it
    # outside a header element.
    text = re.sub(r"^Submitted by admin\s+on\s+[^\n]+\n*", "", text, flags=re.I)
    return text


def fallback_title(page: str, url: str) -> str:
    match = re.search(
        r"<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"']([^\"']+)", page, re.I
    )
    if not match:
        match = re.search(r"<title[^>]*>(.*?)</title>", page, re.I | re.S)
    if match:
        return " ".join(html.unescape(match.group(1)).split()).split(" | Yettel", 1)[0]
    return urlparse(url).path.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title()


def category_for(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    return parts[1] if len(parts) > 1 else "other"


@dataclass
class PageResult:
    index: int
    url: str
    last_modified: str
    html_path: Path
    error: str = ""


def download_one(index: int, item: dict, html_dir: Path, refresh: bool) -> PageResult:
    destination = html_dir / f"{index:04d}.html"
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return PageResult(index, item["url"], item["last_modified"], destination)
    error = ""
    for attempt in range(3):
        try:
            destination.write_bytes(fetch(item["url"]))
            break
        except Exception as exc:  # network failures are recorded in the manifest
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return PageResult(index, item["url"], item["last_modified"], destination, error)


def make_chunks(document: dict, encoding) -> list[dict]:
    token_ids = encoding.encode(document["body"])
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(token_ids):
        end = min(start + CHUNK_TOKENS, len(token_ids))
        text = encoding.decode(token_ids[start:end]).strip()
        chunk_id = f"{document['document_id']}::c{chunk_index:02d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": document["document_id"],
                "title": document["title"],
                "url": document["url"],
                "date": document["date"],
                "category": document["category"],
                "token_start": start,
                "token_end": end,
                "token_count": end - start,
                "text": text,
            }
        )
        if end == len(token_ids):
            break
        start = end - CHUNK_OVERLAP
        chunk_index += 1
    return chunks


def build(args: argparse.Namespace) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_dir = RAW_DIR / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    sitemap_path = RAW_DIR / "sitemap-corporate.xml"
    if args.refresh or not sitemap_path.exists():
        sitemap_path.write_bytes(fetch(SITEMAP_URL))
    inventory = parse_sitemap(sitemap_path.read_bytes())

    results: list[PageResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, index, item, html_dir, args.refresh)
            for index, item in enumerate(inventory, 1)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row.index)

    encoding = tiktoken.get_encoding(TOKENIZER)
    candidates = []
    audit_rows = []
    for result in results:
        if not result.html_path.exists():
            audit_rows.append(
                {"url": result.url, "status": "download_error", "error": result.error}
            )
            continue
        page = result.html_path.read_text(encoding="utf-8", errors="replace")
        extractor = ArticleExtractor()
        extractor.feed(page)
        title = extractor.title or fallback_title(page, result.url)
        body = clean_text(extractor.parts)
        # The title has its own metadata field and must not be duplicated as the
        # first paragraph of the clean body.
        if title and body.startswith(title):
            body = body[len(title) :].lstrip("\n ")
        count = len(encoding.encode(body))
        row = {
            "url": result.url,
            "title": title,
            "category": category_for(result.url),
            "last_modified": result.last_modified,
            "token_count": count,
            "character_count": len(body),
            "status": "eligible"
            if MIN_TOKENS <= count <= MAX_TOKENS
            else "length_excluded",
            "error": result.error,
        }
        audit_rows.append(row)
        if row["status"] == "eligible":
            candidates.append((row, body))

    # Exact clean-body deduplication followed by a deterministic, prose-category
    # prioritisation.  The accepted strict dataset size is 340 documents even if
    # the live sitemap later exposes additional eligible campaign/legal pages.
    seen = set()
    unique_candidates = []
    for row, body in candidates:
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if digest in seen:
            row["status"] = "duplicate_excluded"
            continue
        seen.add(digest)
        unique_candidates.append((row, body, digest))

    category_priority = {
        category: rank
        for rank, category in enumerate(
            [
                "news",
                "faqs",
                "uslugi",
                "planove",
                "rouming",
                "digitalni-uslugi",
                "tv-internet",
                "nashata-mrezha",
                "ustoychivo-razvitie",
                "bezopasen-internet",
                "business",
                "ustroystva",
                "za-nas",
                "karieri",
                "go-plus",
            ]
        )
    }
    selected = sorted(
        unique_candidates,
        key=lambda item: (
            category_priority.get(item[0]["category"], 999),
            abs(item[0]["token_count"] - 2_500),
            item[0]["url"],
        ),
    )[:TARGET_DOCUMENTS]
    selected_urls = {row["url"] for row, _, _ in selected}
    for row, _, _ in unique_candidates:
        if row["url"] not in selected_urls:
            row["status"] = "eligible_not_selected_target_cap"

    # Restore sitemap URL order before assigning stable identifiers.
    selected.sort(
        key=lambda item: next(
            i for i, source in enumerate(inventory) if source["url"] == item[0]["url"]
        )
    )
    documents = []
    for row, body, digest in selected:
        document_id = f"yettel_bg_{len(documents) + 1:04d}"
        documents.append(
            {
                "document_id": document_id,
                "title": row["title"],
                "author": None,
                "source": "Yettel Bulgaria",
                "published_at": None,
                "date": row["last_modified"] or None,
                "date_type": "last_modified" if row["last_modified"] else None,
                "category": row["category"],
                "url": row["url"],
                "language": "bg",
                "tokenizer": TOKENIZER,
                "token_count": row["token_count"],
                "body_sha256": digest,
                "body": body,
            }
        )

    chunks = [
        chunk for document in documents for chunk in make_chunks(document, encoding)
    ]
    write_jsonl(OUT_DIR / "documents.jsonl", documents)
    write_jsonl(OUT_DIR / "chunks_1024.jsonl", chunks)
    write_jsonl(OUT_DIR / "crawl_audit.jsonl", audit_rows)
    (OUT_DIR / "corpus.json").write_text(
        json.dumps(
            [
                {
                    "document_id": d["document_id"],
                    "title": d["title"],
                    "author": d["author"],
                    "source": d["source"],
                    "published_at": d["published_at"],
                    "date": d["date"],
                    "category": d["category"],
                    "url": d["url"],
                    "body": d["body"],
                }
                for d in documents
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    token_counts = [d["token_count"] for d in documents]
    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_sitemap": SITEMAP_URL,
        "source_url_count": len(inventory),
        "document_count": len(documents),
        "eligible_before_target_cap": len(unique_candidates),
        "target_document_count": TARGET_DOCUMENTS,
        "chunk_count": len(chunks),
        "minimum_document_tokens": min(token_counts) if token_counts else None,
        "maximum_document_tokens": max(token_counts) if token_counts else None,
        "mean_document_tokens": round(sum(token_counts) / len(token_counts), 2)
        if token_counts
        else None,
        "selection": {
            "min_tokens": MIN_TOKENS,
            "max_tokens": MAX_TOKENS,
            "tokenizer": TOKENIZER,
        },
        "chunking": {
            "chunk_tokens": CHUNK_TOKENS,
            "overlap_tokens": CHUNK_OVERLAP,
            "tokenizer": TOKENIZER,
        },
        "excluded_content": sorted(ArticleExtractor.SKIP_TAGS),
        "date_note": "Yettel article pages generally omit publication dates; date contains sitemap lastmod and date_type records this explicitly.",
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh", action="store_true", help="Redownload sitemap and HTML cache"
    )
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
