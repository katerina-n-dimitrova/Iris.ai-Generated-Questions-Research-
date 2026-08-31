"""
One-command / staged runner for the MultiHop-RAG dense-vector-only pilot.

    python run_vo.py                       # run the whole pipeline
    python run_vo.py --stage inspect
    python run_vo.py --stage prepare       # select 15 + clean + chunk + align gold
    python run_vo.py --stage generate_questions
    python run_vo.py --stage index         # build the two Chroma collections
    python run_vo.py --stage retrieve
    python run_vo.py --stage evaluate_retrieval
    python run_vo.py --stage evaluate_generation
    python run_vo.py --stage analyze_failures
    python run_vo.py --stage report

Credentials come only from the project .env. NO BM25 / sparse / hybrid anywhere;
retrieval is dense cosine similarity in a local ChromaDB.
"""

from __future__ import annotations

import argparse
import json

import vo_config as C


def stage_inspect():
    import vo_data as D

    corpus = D.load_corpus()
    queries = D.load_all_queries()
    print(
        f"corpus.json      : {len(corpus)} articles (fields: {list(corpus[0].keys())})"
    )
    print(
        f"MultiHopRAG.json : {len(queries)} queries "
        f"(fields: {[k for k in queries[0] if k != 'query_id']})"
    )
    from collections import Counter

    print("question types   :", dict(Counter(q["question_type"] for q in queries)))
    print(f"article id field : {C.ARTICLE_ID_FIELD}")
    print(
        f"embedding model  : env HF_EMBEDDING_MODEL (Octen) | gen model: {C.gen_model()}"
    )
    print(f"chroma store     : LOCAL {C.CHROMA_DIR}")


def stage_prepare():
    import vo_data as D

    print(json.dumps(D.build_all(force=True), indent=2)[:1500])


def stage_generate():
    import vo_generate as G

    G.generate_all()


def stage_index():
    import vo_index as IDX

    IDX.build_indexes()
    print(IDX.verify_ready())


def stage_retrieve():
    import vo_retrieval as R

    R.run_retrieval()


def stage_eval_retrieval():
    import vo_metrics as M

    M.run_metrics()


def stage_eval_generation():
    import vo_answers as A

    A.run_answers()


def stage_failures():
    import vo_failure as F

    F.run_failure_analysis()


def stage_report():
    import vo_report as RP

    RP.build()


STAGES = {
    "inspect": stage_inspect,
    "prepare": stage_prepare,
    "select_subset": stage_prepare,
    "align_gold": stage_prepare,
    "generate_questions": stage_generate,
    "validate_questions": stage_generate,
    "index": stage_index,
    "embed": stage_index,
    "retrieve": stage_retrieve,
    "evaluate_retrieval": stage_eval_retrieval,
    "evaluate_generation": stage_eval_generation,
    "analyze_failures": stage_failures,
    "report": stage_report,
}

_ALL_ORDER = [
    "prepare",
    "generate_questions",
    "index",
    "retrieve",
    "evaluate_retrieval",
    "evaluate_generation",
    "analyze_failures",
    "report",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    if args.stage == "all":
        for s in _ALL_ORDER:
            print(f"\n===== stage: {s} =====")
            STAGES[s]()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
