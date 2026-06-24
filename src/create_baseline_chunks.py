"""
Create the *baseline* processed chunk files for every dataset.

This is a thin orchestrator over the per-dataset preprocess_*.py modules: it
calls each module's build_documents(), keeps only the baseline condition, and
writes processed/<dataset>_baseline.jsonl. It also writes the per-dataset
queries files (shared by both conditions).

Usage:
    python src/create_baseline_chunks.py
    python src/create_baseline_chunks.py --datasets scifact nfcorpus
"""

from __future__ import annotations

import argparse

import common
import config
import preprocess_chartqa
import preprocess_formulareasoning
import preprocess_nfcorpus
import preprocess_scifact
import preprocess_wikitablequestions

MODULES = {
    "scifact": preprocess_scifact,
    "nfcorpus": preprocess_nfcorpus,
    "wikitablequestions": preprocess_wikitablequestions,
    "chartqa": preprocess_chartqa,
    "formulareasoning": preprocess_formulareasoning,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=config.ALL_DATASETS,
                    choices=config.ALL_DATASETS)
    ap.add_argument("--max-samples", type=int, default=config.MAX_DATASET_SAMPLES)
    args = ap.parse_args()

    for name in args.datasets:
        spec = config.DATASETS[name]
        mod = MODULES[name]
        print(f"\n[{name}] building baseline chunks...")
        try:
            docs = mod.build_documents(use_llm=False, max_samples=args.max_samples)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
            continue
        n = common.write_jsonl(spec.processed_path("baseline"), docs["baseline"])
        print(f"  {n} baseline chunks -> {spec.processed_path('baseline').name}")

        queries = mod.build_queries(args.max_samples)
        qpath = config.PROCESSED_DIR / f"{name}_queries.jsonl"
        common.write_jsonl(qpath, queries)
        print(f"  {len(queries)} queries -> {qpath.name}")


if __name__ == "__main__":
    main()
