"""Run the selected GEPA prompt through the complete hierarchical experiment."""

from __future__ import annotations

import json
from pathlib import Path

import vo_hierarchical_hybrid as H

GEPA_DIR = H.RESULTS / "gepa"
BEST_PROMPT = GEPA_DIR / "best_prompt.txt"
GEPA_DOCQ = H.DATA / "gepa_verified_document_questions.jsonl"
GEPA_RANKINGS = GEPA_DIR / "full_rankings.json"
GEPA_VALIDATION = GEPA_DIR / "evidence_validation_answers.jsonl"
ORIGINAL_RANKINGS = H.RANKINGS_PATH
ORIGINAL_METRICS = H.METRICS_PATH


def main():
    if not BEST_PROMPT.exists():
        raise FileNotFoundError(f"Missing optimized prompt: {BEST_PROMPT}")

    # Preserve the original experiment before switching the generation prompt
    # and isolated caches for the GEPA arm.
    original_payload = json.loads(ORIGINAL_RANKINGS.read_text())
    original_conditions = original_payload["conditions"]

    H.GEN_SYSTEM = BEST_PROMPT.read_text()
    H.DOCQ_PATH = GEPA_DOCQ
    H.RANKINGS_PATH = GEPA_RANKINGS
    H.VALIDATION_PATH = GEPA_VALIDATION

    summary = H.prepare(force=False)
    doc_rows = H.generate_document_questions(force=False)
    if not all(x["valid"] and len(x["questions"]) == 10 for x in doc_rows):
        raise RuntimeError("GEPA question generation incomplete; rerun")

    analyses = H.query_understanding(force=False)
    index = H.build_indexes(doc_rows)
    gepa_conditions = H.retrieve_all(index, analyses, force_validation=False)

    merged = dict(original_conditions)
    merged["GEPA-Hier"] = gepa_conditions["Hier-hybrid"]
    merged["GEPA-CE"] = gepa_conditions["Hier-CE"]
    merged["GEPA-final"] = gepa_conditions["Hier-final"]

    ORIGINAL_RANKINGS.write_text(
        json.dumps(
            {
                **original_payload,
                "conditions": merged,
                "gepa_source_rankings": str(GEPA_RANKINGS),
                "gepa_prompt": str(BEST_PROMPT),
            }
        )
    )
    H.METRICS_PATH = ORIGINAL_METRICS
    result = H.evaluate(merged, summary)
    H.render(result)

    for tag in ("GEPA-Hier", "GEPA-CE", "GEPA-final"):
        values = [
            result["metrics"][key][tag]["mean"]
            for key in (
                "evidence_recall@1",
                "evidence_recall@5",
                "evidence_recall@10",
                "all_evidence_hit@5",
                "mrr@10",
            )
        ]
        print(tag, " ".join(f"{x:.3f}" for x in values))


if __name__ == "__main__":
    main()
