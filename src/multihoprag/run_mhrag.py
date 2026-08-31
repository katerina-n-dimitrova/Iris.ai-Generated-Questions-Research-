"""
Umbrella runner: data -> generation -> indexing -> retrieval -> evaluation ->
experiment analyses -> report, for any set of arms.

    python run_mhrag.py                       # baselines B0, B1
    python run_mhrag.py --arms all            # every arm (all 5 experiments + atomic)
    python run_mhrag.py --arms B0 B1 E1       # a subset
    python run_mhrag.py --skip-generate       # reuse cached questions
    python run_mhrag.py --eval-only           # recompute metrics + report from disk

Reproducible + cached: questions/atoms, classifications and rankings are all
cached, so re-runs only do new work. Dependencies between arms (e.g. E4c reusing
B1 + E4b, E6aq needing E6as atoms) are resolved automatically.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import mhrag_config as C
import mhrag_data as D
import mhrag_generate as G
import mhrag_atoms as A
import mhrag_ontology as O
import mhrag_style as S
import mhrag_eval as E
import mhrag_retrieval as R
import mhrag_report as REP

PRODUCING_KINDS = {
    "questions",
    "questions_typed",
    "keywords",
    "qa",
    "fewshot",
    "atoms",
    "atom_questions",
}


def _fmt(ci):
    return f"{ci['mean']:.3f} [{ci['ci_low']:.3f},{ci['ci_high']:.3f}]" if ci else "-"


def _produces(arm_name: str) -> bool:
    a = C.ARMS[arm_name]
    return bool(a.prompt) and a.gen_kind in PRODUCING_KINDS


def resolve_producers(arms: List[str]) -> List[str]:
    """All arms whose generation cache must exist for ``arms`` to run (own gen +
    dense/bm25 sources), ordered so atoms precede atom_questions."""
    need = set()
    for a in arms:
        arm = C.ARMS[a]
        if _produces(a):
            need.add(a)
        for src in (arm.resolved_dense_source(), arm.resolved_bm25_source()):
            if src and _produces(src):
                need.add(src)
    # E6aq depends on E6as atoms
    if "E6aq" in need:
        need.add("E6as")
    order = {"atoms": 0, "atom_questions": 1}
    return sorted(need, key=lambda n: order.get(C.ARMS[n].gen_kind, -1))


def generate_arm(arm_name: str) -> None:
    """Generate an arm's own cache via the right backend (questions / atoms)."""
    chunks = D.load_chunks()
    arm = C.ARMS[arm_name]
    if arm.gen_kind == "atoms":
        gs = A.decompose_for_arm(arm_name, chunks)
    elif arm.gen_kind == "atom_questions":
        gs = A.gen_atom_questions_for_arm(arm_name, chunks, atoms_arm="E6as")
    else:
        gs = G.generate_for_arm(arm_name, chunks)
    print(
        f"[gen:{arm_name}] items={gs.questions_total} avg={gs.avg_questions_per_chunk} "
        f"fail_rate={gs.failure_rate} retried={gs.retried} cost=${gs.estimated_cost_usd}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["B0", "B1"])
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    arms = C.ALL_ARMS if args.arms == ["all"] else args.arms

    t0 = time.perf_counter()
    ds = D.build_all()
    print(
        "[data]",
        json.dumps(
            {
                k: v
                for k, v in ds.items()
                if k not in ("unmatched_evidence", "multi_chunk_evidence")
            }
        ),
    )
    chunks, queries = D.load_chunks(), D.load_queries()

    # Pre-generation setup (only when new generation may run).
    if not args.eval_only and not args.skip_generate:
        if "E1" in arms:
            O.save_allocation(O.allocate_slots())
            print(
                "[types] E1 allocation:",
                {t: c for t, c in O.load_allocation().items() if c},
            )
        if "E5b" in arms:
            info = S.build()
            print(f"[style] {info['n_selected']} exemplars | {info['leakage_check']}")

        # Generate every producing arm (own + reused sources), deps resolved.
        for producer in resolve_producers(arms):
            generate_arm(producer)

    # Retrieval per arm (uses the resolved dense/bm25 sources).
    for arm_name in arms:
        arm = C.ARMS[arm_name]
        if not args.eval_only:
            dense_src = arm.resolved_dense_source()
            bm25_src = arm.resolved_bm25_source()
            dense_q = G.load_questions(dense_src) if dense_src else {}
            bm25_q = G.load_questions(bm25_src) if bm25_src else {}
            rs = R.run_arm(arm_name, chunks, queries, dense_q, bm25_q)
            print(
                f"[index:{arm_name}] vectors={rs.get('num_vectors', '-')} "
                f"(chunk={rs.get('num_chunk_vectors', '-')}, q={rs.get('num_question_vectors', '-')})"
            )
        E.evaluate_arm(arm_name)

    # Experiment-1 cross-tab (needs B1 + E1 classified).
    if not args.eval_only and "E1" in arms and "B1" in arms:
        import mhrag_exp1 as X1

        O.classify_arm_questions("B1", G.load_questions("B1"))
        O.classify_arm_questions("E1", G.load_questions("E1"))
        X1.compute_exp1()

    metrics = {a: E.evaluate_arm(a) for a in arms}

    # --- console summary (overall hybrid) --------------------------------- #
    print("\n" + "=" * 92)
    print("OVERALL — hybrid (dense+BM25 RRF), mean [95% CI]")
    print("=" * 92)
    show = [a for a in arms if a in metrics]
    print(f"{'metric':<20}" + "".join(f"{a:<22}" for a in show))
    for m in E.METRIC_KEYS:
        row = f"{m:<20}"
        for a in show:
            cell = metrics[a]["modes"].get("hybrid", {}).get("overall", {}).get(m)
            row += f"{_fmt(cell):<22}"
        print(row)

    paths = REP.generate(show)
    print(f"\n[done] {round(time.perf_counter() - t0, 1)}s")
    print(f"[report md]   {paths['md']}")
    print(f"[report html] {paths['html']}")


if __name__ == "__main__":
    main()
