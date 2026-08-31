"""
Umbrella runner: data -> generation -> indexing -> retrieval -> evaluation ->
Experiment-1 analysis -> report, for any set of arms.

    python run_qasper.py                       # B0, B1, E1 (+ Exp-1 cross-tab)
    python run_qasper.py --arms B0 B1          # baselines only
    python run_qasper.py --eval-only           # recompute metrics + report from disk

Reproducible + cached: questions, classifications, and rankings are all cached,
so re-runs only do new work. Corpus selection follows QASPER_SELECTION
(default 'text_only' -> the pure-text paper subset).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import qasper_config as C
import qasper_data as D
import qasper_generate as G
import qasper_ontology as O
import qasper_retrieval as R
import qasper_eval as E
import qasper_exp1 as X1
import qasper_report as REP

DEFAULT_ARMS = ["B0", "B1", "E1"]


def _fmt(ci):
    return f"{ci['mean']:.3f} [{ci['ci_low']:.3f},{ci['ci_high']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    arms = args.arms

    t0 = time.perf_counter()
    ds = D.build_all()
    print("[data]", json.dumps(ds))
    chunks, queries = D.load_chunks(), D.load_queries()

    # Query type classification + E1 slot allocation (needed before E1 generation).
    if not args.eval_only:
        qtypes = O.classify_queries(queries)
        alloc = O.allocate_slots(Counter(qtypes.values()))
        O.save_allocation(alloc)
        print("[types] query dist:", dict(Counter(qtypes.values())))
        print("[types] E1 allocation:", {t: c for t, c in alloc.items() if c})

    metrics = {}
    for arm_name in arms:
        arm = C.ARMS[arm_name]
        questions = {}
        if arm.kind == "enrichment":
            if not args.eval_only and not args.skip_generate:
                gs = G.generate_for_arm(arm_name, chunks)
                print(
                    f"[gen:{arm_name}] q_total={gs.questions_total} "
                    f"fail_rate={gs.failure_rate} cost=${gs.estimated_cost_usd}"
                )
            questions = G.load_questions(arm_name)
        if not args.eval_only:
            rs = R.run_arm(arm_name, chunks, queries, questions)
            print(
                f"[index:{arm_name}] vectors={rs['num_vectors']} "
                f"size={rs['index_size_mb']}MB"
            )
        metrics[arm_name] = E.evaluate_arm(arm_name)

    # Classify generated questions (for the Experiment-1 cross-tab).
    if not args.eval_only:
        for a in ("B1", "E1"):
            if a in arms:
                O.classify_arm_questions(a, G.load_questions(a))
    if "E1" in arms:
        X1.compute_exp1()

    # --- console summary (overall hybrid) --------------------------------- #
    print("\n" + "=" * 74)
    print("OVERALL — hybrid (dense+BM25 RRF), mean [95% CI]")
    print("=" * 74)
    print(f"{'metric':<12}" + "".join(f"{a:<24}" for a in arms))
    for m in E.METRIC_KEYS:
        print(
            f"{m:<12}"
            + "".join(
                f"{_fmt(metrics[a]['modes']['hybrid']['overall'][m]):<24}" for a in arms
            )
        )

    paths = REP.generate(arms)
    print(f"\n[done] {round(time.perf_counter() - t0, 1)}s")
    print(f"[report] {paths['html']}")


if __name__ == "__main__":
    main()
