"""
Preprocess FormulaReasoning (formula-based numerical reasoning).

This is the 5th document type: physics *formulas*. Unlike the other QA datasets
we don't self-retrieve — the natural knowledge base is the formula database
(formulas.json, 272 formulas), and each question must retrieve the formula(s)
it needs to compute its numeric answer (the same RAG setup the paper uses).

Documents : one per formula (English form + variables).
Queries   : English physics questions; gold_answer is the numeric answer
            (e.g. "0.3 kg") so full answer-quality metrics apply.
Retrieval gold : derived heuristically — a formula is "relevant" to a question
            when every one of its variables (by English name) appears among the
            question's annotated quantities (`arguments`). ~65% of questions get
            at least one gold formula; only those are used for retrieval eval.

Baseline chunk : the raw formula + its variable names.
Enriched chunk : dataset/type, the formula, a readable variable list, keywords.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Tuple

import common
import config
from common import make_record, render_enriched

SPEC = config.DATASETS["formulareasoning"]


def _load(name: str):
    path = SPEC.raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run download_datasets.py first.")
    return json.load(path.open(encoding="utf-8"))


def _formula_vars(f: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return [(en_name, en_symbol), ...] for a formula's variables."""
    out = []
    for v in f.get("symbol_map", []):
        nm = str(v.get("en_name", "")).strip()
        sym = str(v.get("en_symbol", "")).strip()
        if nm:
            out.append((nm, sym))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _gold_for_question(args: List[Dict[str, Any]],
                       formula_vars: List[Tuple[str, List[str]]]) -> List[str]:
    """A formula is relevant if every variable name appears in the question's
    annotated quantity names."""
    qtext = " ; ".join(_norm(a.get("en_name", "")) for a in args)
    gold = []
    for key, names in formula_vars:
        if names and all(n in qtext for n in names):
            gold.append(key)
    return gold


def build_documents(use_llm: bool = False, max_samples: int = config.MAX_DATASET_SAMPLES
                    ) -> Dict[str, List[Dict[str, Any]]]:
    client = config.get_openai_client() if use_llm else None
    formulas = _load("formulas.json")

    baseline: List[Dict[str, Any]] = []
    enriched: List[Dict[str, Any]] = []

    for f in formulas:
        key = str(f["key"])
        formula_en = str(f.get("formula", {}).get("en", "")).strip()
        if not formula_en:
            continue
        variables = _formula_vars(f)
        var_names = [nm for nm, _ in variables]
        var_readable = ", ".join(f"{nm} ({sym})" if sym else nm
                                 for nm, sym in variables)

        base_text = formula_en + (f"  | variables: {', '.join(var_names)}"
                                  if var_names else "")
        baseline.append(make_record(
            chunk_id=f"formula_{key}_baseline",
            dataset="formulareasoning",
            input_type=SPEC.input_type,
            condition="baseline",
            text_for_embedding=base_text,
            original_text=formula_en,
            source_id=key,
            title=formula_en,
        ))

        summary = common.llm_summary(client, f"Physics formula {formula_en} "
                                     f"with variables {var_readable}",
                                     kind="physics formula") \
            if use_llm else ", ".join(var_names)
        enriched_text = render_enriched({
            "Dataset": "FormulaReasoning",
            "Document type": "physics formula",
            "Formula": formula_en,
            "Variables": var_readable,
            "Keywords / quantities": summary,
        })
        enriched.append(make_record(
            chunk_id=f"formula_{key}_enriched",
            dataset="formulareasoning",
            input_type=SPEC.input_type,
            condition="enriched",
            text_for_embedding=enriched_text,
            original_text=formula_en,
            source_id=key,
            title=formula_en,
        ))

    return {"baseline": baseline, "enriched": enriched}


def build_queries(max_samples: int = config.MAX_DATASET_SAMPLES) -> List[Dict[str, Any]]:
    formulas = _load("formulas.json")
    tests = _load("HeF_test.json")

    formula_vars = [(str(f["key"]),
                     [_norm(nm) for nm, _ in _formula_vars(f)])
                    for f in formulas]

    queries: List[Dict[str, Any]] = []
    for t in tests:
        q = t.get("question", {})
        text = str(q.get("en") or q.get("zh") or "").strip()
        if not text:
            continue
        gold = _gold_for_question(t.get("arguments", []), formula_vars)
        if not gold:
            continue  # keep only queries with a derivable gold formula
        queries.append({
            "query_id": str(t.get("id")),
            "dataset": "formulareasoning",
            "text": text,
            "gold_source_ids": gold,
            "gold_answer": str(t.get("answer", "")),
        })
        if len(queries) >= max_samples:
            break
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--max-samples", type=int, default=config.MAX_DATASET_SAMPLES)
    args = ap.parse_args()

    docs = build_documents(args.use_llm, args.max_samples)
    for condition in config.CONDITIONS:
        n = common.write_jsonl(SPEC.processed_path(condition), docs[condition])
        print(f"formulareasoning {condition}: {n} chunks "
              f"-> {SPEC.processed_path(condition).name}")

    queries = build_queries(args.max_samples)
    qpath = config.PROCESSED_DIR / "formulareasoning_queries.jsonl"
    common.write_jsonl(qpath, queries)
    print(f"formulareasoning queries: {len(queries)} -> {qpath.name}")


if __name__ == "__main__":
    main()
