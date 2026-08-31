"""
Create the *enriched* processed chunk files for every dataset.

Thin orchestrator over the per-dataset preprocess_*.py modules: calls each
module's build_documents(), keeps only the enriched condition, and writes
processed/<dataset>_enriched.jsonl.

By default enrichment uses cheap, offline heuristics (keyword extraction +
extractive summaries) so the pipeline runs without spending API credits.
Pass --use-llm to generate LLM summaries via the configured chat model.

Usage:
    python src/create_enriched_chunks.py
    python src/create_enriched_chunks.py --use-llm
    python src/create_enriched_chunks.py --datasets scifact --use-llm
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
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=config.ALL_DATASETS,
        choices=config.ALL_DATASETS,
    )
    ap.add_argument(
        "--use-llm",
        action="store_true",
        help="Generate enrichment summaries with the OpenAI chat model.",
    )
    ap.add_argument("--max-samples", type=int, default=config.MAX_DATASET_SAMPLES)
    args = ap.parse_args()

    for name in args.datasets:
        spec = config.DATASETS[name]
        mod = MODULES[name]
        print(f"\n[{name}] building enriched chunks (use_llm={args.use_llm})...")
        try:
            docs = mod.build_documents(
                use_llm=args.use_llm, max_samples=args.max_samples
            )
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
            continue
        n = common.write_jsonl(spec.processed_path("enriched"), docs["enriched"])
        print(f"  {n} enriched chunks -> {spec.processed_path('enriched').name}")


if __name__ == "__main__":
    main()
