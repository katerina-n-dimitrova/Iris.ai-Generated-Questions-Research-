"""
Run the two baselines end-to-end (B0 + B1) and report their metrics.

This validates the whole harness before the five question-type experiments:
data -> (generation for B1) -> dense+BM25 indexing -> RRF retrieval -> metrics
with bootstrap CIs, broken out by retrieval mode and QASPER answer type.

    python run_qasper_baselines.py                 # both baselines
    python run_qasper_baselines.py --arms B0       # one
    python run_qasper_baselines.py --skip-generate # assume B1 questions cached
    python run_qasper_baselines.py --eval-only     # recompute metrics from disk

Everything is cached (questions, and rankings on disk), so re-runs are cheap.
"""

from __future__ import annotations

import argparse
import json
import time

import qasper_config as C
import qasper_data as D
import qasper_generate as G
import qasper_retrieval as R
import qasper_eval as E


def _fmt(ci: dict) -> str:
    return f"{ci['mean']:.3f} [{ci['ci_low']:.3f},{ci['ci_high']:.3f}]"


def _print_overall_table(metrics_by_arm: dict) -> None:
    arms = list(metrics_by_arm.keys())
    print("\n" + "=" * 78)
    print("OVERALL (hybrid = dense+BM25 RRF) — mean [95% bootstrap CI]")
    print("=" * 78)
    header = f"{'metric':<12}" + "".join(f"{a:<26}" for a in arms)
    print(header)
    for key in E.METRIC_KEYS:
        row = f"{key:<12}"
        for a in arms:
            row += f"{_fmt(metrics_by_arm[a]['modes']['hybrid']['overall'][key]):<26}"
        print(row)


def _print_mode_table(metrics_by_arm: dict) -> None:
    print("\n" + "=" * 78)
    print("WHERE GAINS COME FROM — Recall@10 by retrieval mode")
    print("=" * 78)
    arms = list(metrics_by_arm.keys())
    print(f"{'mode':<10}" + "".join(f"{a:<26}" for a in arms))
    for mode in C.RETRIEVAL_MODES:
        row = f"{mode:<10}"
        for a in arms:
            row += (
                f"{_fmt(metrics_by_arm[a]['modes'][mode]['overall']['recall@10']):<26}"
            )
        print(row)


def _print_type_table(metrics_by_arm: dict) -> None:
    print("\n" + "=" * 78)
    print("PER ANSWER TYPE — Recall@10 (hybrid)")
    print("=" * 78)
    arms = list(metrics_by_arm.keys())
    print(f"{'type':<14}" + "".join(f"{a:<26}" for a in arms))
    for atype in C.ANSWER_TYPES:
        row = f"{atype:<14}"
        for a in arms:
            row += f"{_fmt(metrics_by_arm[a]['modes']['hybrid']['by_answer_type'][atype]['recall@10']):<26}"
        print(row)


def _print_significance(scored_cache: dict) -> None:
    print("\n" + "=" * 78)
    print("SIGNIFICANCE — B1 minus B0 (hybrid), paired bootstrap")
    print("=" * 78)
    if "B0" not in scored_cache or "B1" not in scored_cache:
        print("  (need both B0 and B1)")
        return
    for key in E.METRIC_KEYS:
        d = E.paired_delta(scored_cache["B1"], scored_cache["B0"], key)
        star = " *SIG*" if d["significant"] else ""
        print(
            f"  {key:<12} Δ={d['delta']:+.3f} "
            f"[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] p={d.get('p_value')}{star}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=C.BASELINE_ARMS)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="Recompute metrics from saved rankings only.",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    ds = D.build_all()
    print("[data]", json.dumps(ds))
    chunks = D.load_chunks()
    queries = D.load_queries()

    run_summary = {"config": C.run_config_signature(), "dataset": ds, "arms": {}}
    metrics_by_arm, scored_cache = {}, {}

    for arm_name in args.arms:
        arm = C.ARMS[arm_name]
        questions = {}
        if arm.kind == "enrichment":
            if not args.skip_generate and not args.eval_only:
                gstats = G.generate_for_arm(arm_name, chunks)
                run_summary["arms"].setdefault(arm_name, {})["generation"] = (
                    gstats.__dict__
                )
                print(f"[gen:{arm_name}]", json.dumps(gstats.__dict__))
            questions = G.load_questions(arm_name)

        if not args.eval_only:
            rstats = R.run_arm(arm_name, chunks, queries, questions)
            run_summary["arms"].setdefault(arm_name, {})["retrieval"] = rstats
            print(
                f"[index:{arm_name}]",
                json.dumps(rstats["latency_ms"]),
                f"vectors={rstats['num_vectors']} size={rstats['index_size_mb']}MB",
            )

        metrics_by_arm[arm_name] = E.evaluate_arm(arm_name)
        scored_cache[arm_name] = E.load_scored(arm_name, "hybrid")

    _print_overall_table(metrics_by_arm)
    _print_mode_table(metrics_by_arm)
    _print_type_table(metrics_by_arm)
    _print_significance(scored_cache)

    run_summary["wall_seconds"] = round(time.perf_counter() - t0, 1)
    with (C.RESULTS_DIR / "baselines_run_summary.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(run_summary, fh, indent=2)
    print(
        f"\n[done] {run_summary['wall_seconds']}s — "
        f"summary: {C.RESULTS_DIR / 'baselines_run_summary.json'}"
    )


if __name__ == "__main__":
    main()
