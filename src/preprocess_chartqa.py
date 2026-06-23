"""
Preprocess ChartQA (charts / graphs).

NOTE ON LIMITATIONS
-------------------
The HuggingFaceM4/ChartQA rows are (image, query, label). There is no bundled
OCR / caption / axis-label text, so the *purely textual* content available
without a vision model is limited. For this first end-to-end version we treat
each chart's answer label as its textual content stand-in and leave explicit
placeholder fields (chart type, axis labels, legend) in the enriched format.

To get real signal, plug a multimodal OCR / chart-understanding step into
`extract_chart_text()` below (e.g. an OpenAI vision call on the image, or a
chart-to-table model). The rest of the pipeline does not need to change.

Produces document records (baseline + enriched) and a queries file from
data/raw/chartqa/.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import common
import config
from common import make_record, read_jsonl, render_enriched

SPEC = config.DATASETS["chartqa"]


def extract_chart_text(row: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Pull whatever textual signal we can from a ChartQA row. Returns a dict of
    optional fields. Extend this with a vision/OCR model for real chart text.
    """
    return {
        "ocr_text": str(row.get("label") or "").strip(),  # answer value as stand-in
        "title": None,
        "chart_type": None,
        "x_axis": None,
        "y_axis": None,
        "legend": None,
    }


def build_documents(use_llm: bool = False, max_samples: int = config.MAX_DATASET_SAMPLES
                    ) -> Dict[str, List[Dict[str, Any]]]:
    client = config.get_openai_client() if use_llm else None
    test_path = SPEC.raw_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} missing. Run download_datasets.py first.")

    baseline: List[Dict[str, Any]] = []
    enriched: List[Dict[str, Any]] = []

    for idx, row in enumerate(list(read_jsonl(test_path))[:max_samples]):
        chart_id = f"chartqa_{idx}"
        fields = extract_chart_text(row)
        ocr = fields["ocr_text"] or ""
        base_text = (fields["title"] or "") + (" " if fields["title"] else "") + ocr
        base_text = base_text.strip() or ocr or "(no extractable chart text)"

        baseline.append(make_record(
            chunk_id=f"{chart_id}_baseline",
            dataset="chartqa",
            input_type=SPEC.input_type,
            condition="baseline",
            text_for_embedding=base_text,
            original_text=base_text,
            source_id=chart_id,
            title=fields["title"],
            chart_id=chart_id,
        ))

        summary = common.llm_summary(client, base_text, kind="chart") \
            if use_llm else common.cheap_summary(base_text)
        enriched_text = render_enriched({
            "Dataset": "ChartQA",
            "Document type": "chart",
            "Chart title": fields["title"],
            "Chart type if available": fields["chart_type"],
            "X-axis label if available": fields["x_axis"],
            "Y-axis label if available": fields["y_axis"],
            "Legend if available": fields["legend"],
            "Generated chart summary": summary,
            "Original chart text/caption/OCR": base_text,
        })
        enriched.append(make_record(
            chunk_id=f"{chart_id}_enriched",
            dataset="chartqa",
            input_type=SPEC.input_type,
            condition="enriched",
            text_for_embedding=enriched_text,
            original_text=base_text,
            source_id=chart_id,
            title=fields["title"],
            chart_id=chart_id,
        ))

    return {"baseline": baseline, "enriched": enriched}


def build_queries(max_samples: int = config.MAX_DATASET_SAMPLES) -> List[Dict[str, Any]]:
    test_path = SPEC.raw_dir / "test.jsonl"
    if not test_path.exists():
        return []
    queries: List[Dict[str, Any]] = []
    for idx, row in enumerate(list(read_jsonl(test_path))[:max_samples]):
        chart_id = f"chartqa_{idx}"
        question = str(row.get("query") or row.get("question") or "").strip()
        if not question:
            continue
        queries.append({
            "query_id": f"chartqa_q{idx}",
            "dataset": "chartqa",
            "text": question,
            "gold_source_ids": [chart_id],
            "gold_answer": str(row.get("label") or ""),
        })
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--max-samples", type=int, default=config.MAX_DATASET_SAMPLES)
    args = ap.parse_args()

    docs = build_documents(args.use_llm, args.max_samples)
    for condition in config.CONDITIONS:
        n = common.write_jsonl(SPEC.processed_path(condition), docs[condition])
        print(f"chartqa {condition}: {n} chunks -> {SPEC.processed_path(condition).name}")

    queries = build_queries(args.max_samples)
    qpath = config.PROCESSED_DIR / "chartqa_queries.jsonl"
    common.write_jsonl(qpath, queries)
    print(f"chartqa queries: {len(queries)} -> {qpath.name}")


if __name__ == "__main__":
    main()
