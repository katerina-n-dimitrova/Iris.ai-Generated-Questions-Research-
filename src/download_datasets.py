"""
Download the four experiment datasets from the Hugging Face Hub and cache a
small subset locally as raw JSONL.

We do NOT download full datasets eagerly: each dataset is streamed/truncated to
MAX_DATASET_SAMPLES (configurable via .env or --max-samples). Raw dumps land in
data/raw/<dataset>/.

Usage:
    python src/download_datasets.py                  # all datasets
    python src/download_datasets.py --datasets scifact nfcorpus
    python src/download_datasets.py --max-samples 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

import config


def _save_split(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"  wrote {len(records):>5} rows -> {path.relative_to(config.PROJECT_ROOT)}")


def _take(dataset_iterable, n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(dataset_iterable):
        if i >= n:
            break
        rows.append(dict(ex))
    return rows


def download_one(name: str, max_samples: int) -> None:
    spec = config.DATASETS[name]
    print(f"\n[{name}] {spec.hf_id}"
          + (f" :: {spec.hf_config}" if spec.hf_config else ""))
    out_dir = spec.raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # FormulaReasoning lives on GitHub (not the HF Hub): fetch the formula
    # database + an English test split via raw URLs.
    if name == "formulareasoning":
        import urllib.request
        base = "https://raw.githubusercontent.com/nju-websoft/FormulaReasoning/main"
        for rel, dest in [
            ("formulas.json", "formulas.json"),
            ("data/FormulaReasoning/HeF_test.json", "HeF_test.json"),
        ]:
            target = out_dir / dest
            urllib.request.urlretrieve(f"{base}/{rel}", target)
            print(f"  fetched {rel} -> {target.relative_to(config.PROJECT_ROOT)}")
        return

    # ChartQA stores images; we only need the textual fields, so we drop the
    # image column on load to keep the raw cache small and JSON-serialisable.
    if name == "chartqa":
        ds = load_dataset(spec.hf_id, split="test")
        cols_to_drop = [c for c in ds.column_names if c == "image"]
        if cols_to_drop:
            ds = ds.remove_columns(cols_to_drop)
        rows = _take(ds, max_samples)
        _save_split(rows, out_dir / "test.jsonl")
        return

    if name == "wikitablequestions":
        ds = load_dataset(spec.hf_id, split="test")
        rows = _take(ds, max_samples)
        _save_split(rows, out_dir / "test.jsonl")
        return

    # scifact / nfcorpus: BEIR-style corpus + queries + qrels.
    # We pull the corpus (documents to index) plus queries + qrels for eval.
    corpus = load_dataset(spec.hf_id, "corpus", split="corpus")
    _save_split(_take(corpus, max_samples), out_dir / "corpus.jsonl")

    try:
        queries = load_dataset(spec.hf_id, "queries", split="queries")
        _save_split(_take(queries, max_samples), out_dir / "queries.jsonl")
    except Exception as e:  # noqa: BLE001
        print(f"  (no queries split: {e})")

    # qrels live in the parent BEIR repo, e.g. BeIR/nfcorpus-qrels
    qrels_id = f"{spec.hf_id}-qrels"
    try:
        for split in ("test", "validation", "train"):
            try:
                qrels = load_dataset(qrels_id, split=split)
            except Exception:
                continue
            _save_split(list(qrels), out_dir / f"qrels_{split}.jsonl")
    except Exception as e:  # noqa: BLE001
        print(f"  (no qrels for {qrels_id}: {e})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=config.ALL_DATASETS,
                    choices=config.ALL_DATASETS,
                    help="Which datasets to download (default: all).")
    ap.add_argument("--max-samples", type=int, default=config.MAX_DATASET_SAMPLES,
                    help="Cap rows per split (default from .env).")
    args = ap.parse_args()

    print(f"Downloading {len(args.datasets)} dataset(s), "
          f"max {args.max_samples} samples each.")
    for name in tqdm(args.datasets, desc="datasets"):
        try:
            download_one(name, args.max_samples)
        except Exception as e:  # noqa: BLE001
            print(f"  !! failed to download {name}: {e}")
    print("\nDone. Raw data in", config.RAW_DIR.relative_to(config.PROJECT_ROOT))


if __name__ == "__main__":
    main()
