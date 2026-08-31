"""
Staged runner for the 10-article atomic+chunk-level mixed-question pilot.

    python run_am.py                          # whole pipeline
    python run_am.py --stage inspect
    python run_am.py --stage prepare          # select 10 + clean + chunk + gold
    python run_am.py --stage generate         # atoms + atomic + chunk-level questions
    python run_am.py --stage filter           # validate + round-trip + margin + coverage
    python run_am.py --stage index            # baseline + pooled question collections
    python run_am.py --stage retrieve
    python run_am.py --stage evaluate_retrieval
    python run_am.py --stage diagnostics
    python run_am.py --stage evaluate_generation
    python run_am.py --stage analyze_failures
    python run_am.py --stage report

Dense-vector-only. Credentials only from .env. Isolated namespace so the
15-article mhrag_vectoronly experiment stays intact.
"""

from __future__ import annotations

import argparse

import am_config as C


def s_inspect():
    import am_data as D

    corpus = D.load_corpus()
    queries = D.load_all_queries()
    from collections import Counter

    print(
        f"corpus {len(corpus)} articles | queries {len(queries)} "
        f"({dict(Counter(q['question_type'] for q in queries))})"
    )
    print(
        f"article budget {C.ARTICLE_COUNT} | chunk {C.CHUNK_SIZE}/{C.CHUNK_OVERLAP}/{C.MIN_CHUNK} | "
        f"gen model {C.gen_model()} | chroma LOCAL {C.CHROMA_DIR}"
    )


_FORCE = False


def s_prepare():
    import am_data as D
    import json

    print(json.dumps(D.build_all(force=_FORCE), indent=2)[:1200])


def s_generate():
    import am_generate as G

    G.generate_all(force=_FORCE)


def s_filter():
    import am_filter as F

    F.run_filter(force=_FORCE)


def s_index():
    import am_index as I

    I.build_all()
    print(I.verify_ready())


def s_retrieve():
    import am_retrieval as R

    R.run_retrieval()


def s_eval_ret():
    import am_metrics as M

    M.run_metrics()


def s_diag():
    import am_diagnostics as Dg

    Dg.run_diagnostics()


def s_eval_gen():
    import am_answers as A

    A.run_answers()


def s_fail():
    import am_failure as F

    F.run_failure_analysis()


def s_report():
    import am_report as RP

    RP.build()


STAGES = {
    "inspect": s_inspect,
    "prepare": s_prepare,
    "generate": s_generate,
    "filter": s_filter,
    "index": s_index,
    "retrieve": s_retrieve,
    "evaluate_retrieval": s_eval_ret,
    "diagnostics": s_diag,
    "evaluate_generation": s_eval_gen,
    "analyze_failures": s_fail,
    "report": s_report,
}
_ORDER = [
    "prepare",
    "generate",
    "filter",
    "index",
    "retrieve",
    "evaluate_retrieval",
    "diagnostics",
    "evaluate_generation",
    "analyze_failures",
    "report",
]


def main():
    global _FORCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild cached artifacts for the selected stage(s)",
    )
    a = ap.parse_args()
    _FORCE = a.force
    if a.stage == "all":
        for s in _ORDER:
            print(f"\n===== {s} =====")
            STAGES[s]()
    else:
        STAGES[a.stage]()


if __name__ == "__main__":
    main()
