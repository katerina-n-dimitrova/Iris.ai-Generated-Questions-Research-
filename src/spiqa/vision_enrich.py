"""
Vision enrichment for SPIQA figure/chart/plot/table images.

SPIQA ships figures as images; until now every figure/table retrieval unit was
represented only by its caption + type + nearby text. Here we add a VISUAL
DESCRIPTION paragraph: a vision model (gpt-4o-mini, the same model family used for
question generation — vision-capable, uses the existing OPENAI_API_KEY, no new
provider) looks at the actual image and writes a factual, grounded paragraph of
what it shows (chart type, axes, trends, notable values, entities). That
description is folded into the figure unit's context enrichment so retrieval can
match questions about the visual content, not just the caption.

Grounding rules baked into the prompt: describe only what is visible; read axis
labels / legends / values when legible; do not invent numbers. Cached to disk
(resumable) keyed by fig_id so images are never re-described.
"""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import spiqa_config as C
from question_gen import estimate_cost

IMAGES_ROOT = C.RAW_DIR / "testB_images" / "SPIQA_testB_Images_224px"
CACHE = C.PROCESSED_DIR / "test-B_vision_captions.jsonl"
VISION_MODEL = __import__("os").getenv("SPIQA_VISION_MODEL", "gpt-4o-mini")


def image_path(fig_id: str) -> Optional[Path]:
    p = IMAGES_ROOT / fig_id
    return p if p.exists() else None


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _prompt(caption: str, kind: str, fig_type: str) -> str:
    return (
        f"This is a {fig_type or kind} from a scientific paper. "
        f'Caption: "{caption or "n/a"}".\n'
        "Write ONE factual paragraph (<=80 words) describing what the image "
        "actually shows. If it is a plot/chart: state the chart type, what the "
        "x and y axes represent, the series/legend, and the main trend or notable "
        "values that are legible. If it is a table: state what it compares and the "
        "row/column headers. If it is a schematic/diagram: state the components and "
        "their relationships. Describe ONLY what is visible; read labels/values "
        "when legible; never invent numbers or entities."
    )


def caption_one(client, fig_id: str, caption: str, kind: str, fig_type: str):
    path = image_path(fig_id)
    if path is None:
        return None, 0, 0
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0.2,
        max_tokens=180,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _prompt(caption, kind, fig_type)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_b64(path)}",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ],
    )
    u = resp.usage
    return (
        (resp.choices[0].message.content or "").strip(),
        getattr(u, "prompt_tokens", 0) or 0,
        getattr(u, "completion_tokens", 0) or 0,
    )


def _load_cache() -> Dict[str, dict]:
    cache = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cache[r["fig_id"]] = r
    return cache


def generate_vision(
    figure_units: List[Dict],
    *,
    max_workers: int = C.LLM_MAX_WORKERS,
    limit: Optional[int] = None,
) -> Dict:
    """
    figure_units: list of dicts with keys fig_id, caption, kind, fig_type.
    Describes each image (cached, resumable). Returns a cost/time summary.
    """
    import config

    client = config.get_openai_client()
    cache = _load_cache()
    todo = [
        f for f in figure_units if f["fig_id"] not in cache and image_path(f["fig_id"])
    ]
    if limit:
        todo = todo[:limit]

    pt = ct = 0
    t0 = time.perf_counter()
    fh = CACHE.open("a", encoding="utf-8")

    def work(f):
        desc, p, c = caption_one(
            client,
            f["fig_id"],
            f.get("caption", ""),
            f.get("kind", "figure"),
            f.get("fig_type", ""),
        )
        return f, desc, p, c

    try:
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(work, f): f for f in todo}
            for fut in as_completed(futs):
                f, desc, p, c = fut.result()
                pt += p
                ct += c
                if desc is not None:
                    row = {
                        "fig_id": f["fig_id"],
                        "kind": f.get("kind"),
                        "fig_type": f.get("fig_type"),
                        "description": desc,
                        "prompt_tokens": p,
                        "completion_tokens": c,
                    }
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    cache[f["fig_id"]] = row
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(
                        f"  [vision] {done}/{len(todo)} (cache {len(cache)})",
                        flush=True,
                    )
    finally:
        fh.close()
    return {
        "newly_described": len(todo),
        "cached_now": len(cache),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "estimated_cost_usd": estimate_cost(pt, ct, VISION_MODEL),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }


def load_vision() -> Dict[str, str]:
    """Return {fig_id: description}."""
    out = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("description"):
                    out[r["fig_id"]] = r["description"]
    return out


def figure_units_from_chunks(chunks: List[Dict]) -> List[Dict]:
    """Extract the figure/table units (with their fig_id/caption/type) from chunks."""
    units = []
    for c in chunks:
        if c["source_type"] in ("figure_caption", "table_caption"):
            m = c["metadata"]
            units.append(
                {
                    "chunk_id": c["chunk_id"],
                    "fig_id": m.get("fig_id", ""),
                    "caption": m.get("caption", ""),
                    "kind": m.get("fig_kind", "figure"),
                    "fig_type": m.get("fig_type", ""),
                }
            )
    return units


def apply_vision_to_chunks(chunks: List[Dict], vision: Dict[str, str]) -> List[Dict]:
    """
    Return a NEW chunk list where figure/table units have the visual description
    appended to their text. Text chunks are unchanged.
    """
    out = []
    for c in chunks:
        c2 = dict(c)
        c2["metadata"] = dict(c["metadata"])
        if c["source_type"] in ("figure_caption", "table_caption"):
            fid = c["metadata"].get("fig_id", "")
            desc = vision.get(fid)
            if desc:
                c2["text"] = c["text"] + f"\nVisual description: {desc}"
                c2["metadata"]["has_vision"] = True
        out.append(c2)
    return out


if __name__ == "__main__":
    import argparse
    from spiqa_loader import load_papers
    from spiqa_chunker import load_chunks

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    papers = load_papers("test-B", args.max_papers)
    keep = {p.paper_id for p in papers}
    chunks = [c for c in load_chunks("test-B") if c["paper_id"] in keep]
    units = figure_units_from_chunks(chunks)
    print(f"figure/table units in {len(papers)} papers: {len(units)}")
    print(json.dumps(generate_vision(units, limit=args.limit), indent=2))
