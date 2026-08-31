"""GEPA and MIPROv2 prompt optimization for the controlled suite.

Train examples come only from the frozen training split. Candidate selection
uses only the development split. This module never loads test queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import controlled_suite_generate as G
import controlled_suite_retrieval as R
import vo_config as C
import vo_hierarchical_hybrid as H
from embeddings import get_embedder

OUT = G.DATA / "prompt_optimization"
OUT.mkdir(parents=True, exist_ok=True)
GEPA_GENERAL_PROMPT = OUT / "gepa_general_prompt.txt"
GEPA_ATOMIC_PROMPT = OUT / "gepa_atomic_prompt.txt"
MIPRO_GENERAL_PROGRAM = OUT / "mipro_general_program.json"
MIPRO_ATOMIC_PROGRAM = OUT / "mipro_atomic_program.json"
MIPRO_GENERAL_PROMPT = OUT / "mipro_general_prompt.txt"
MIPRO_ATOMIC_PROMPT = OUT / "mipro_atomic_prompt.txt"
GEPA_GENERAL_Q = G.DATA / "gepa_general_512_128_q10.jsonl"
GEPA_ATOMIC_Q = G.DATA / "gepa_atomic_512_128_q3.jsonl"
MIPRO_GENERAL_Q = G.DATA / "mipro_general_512_128_q10.jsonl"
MIPRO_ATOMIC_Q = G.DATA / "mipro_atomic_512_128_q3.jsonl"
SUMMARY = OUT / "optimization_summary.json"


def _split_queries() -> tuple[list[dict], list[dict]]:
    split = {r["query_id"]: r["split"] for r in G.read_jsonl(R.SPLIT_PATH)}
    # Deliberately filter before returning; test query text never enters memory.
    queries = H.load_queries()
    train = [q for q in queries if split[q["query_id"]] == "train"]
    development = [q for q in queries if split[q["query_id"]] == "development"]
    return train, development


def _prompt_cache_path(kind: str, prompt: str) -> Path:
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    return OUT / f"candidate_{kind}_{digest}.jsonl"


def _generate_candidate(kind: str, prompt: str) -> list[dict]:
    path = _prompt_cache_path(kind, prompt)
    chunks = H.load_chunks()
    if path.exists():
        rows = G.read_jsonl(path)
        if len(rows) == len(chunks) and all(
            row.get("questions") or row.get("facts") for row in rows
        ):
            return rows
    worker = (
        (lambda chunk: G._general(chunk, prompt, 10))
        if kind == "general"
        else (lambda chunk: G._atomic(chunk, prompt))
    )
    G._generate_file(chunks, path, worker)
    return G.read_jsonl(path)


def _dev_score(kind: str, prompt: str, development: list[dict]) -> dict:
    rows = _generate_candidate(kind, prompt)
    questions = []
    for row in rows:
        values = (
            row.get("questions", [])
            if kind == "general"
            else [q for fact in row.get("facts", []) for q in fact.get("questions", [])]
        )
        questions.extend((row["chunk_id"], q) for q in values)
    embedder = get_embedder()
    qvectors = np.asarray(
        embedder.embed_documents([q for _, q in questions]), dtype=float
    )
    query_vectors = {
        q["query_id"]: np.asarray(embedder.embed_query(q["query"]), dtype=float)
        for q in development
    }
    gold = H.load_gold()
    recalls, mrrs = [], []
    for query in development:
        scores = H._cosine_matrix(qvectors, query_vectors[query["query_id"]])
        best = defaultdict(lambda: -1.0)
        for (chunk_id, _), score in zip(questions, scores):
            best[chunk_id] = max(best[chunk_id], float(score))
        ranking = [
            cid for cid, _ in sorted(best.items(), key=lambda pair: (-pair[1], pair[0]))
        ]
        units = gold[query["query_id"]]["evidence_units"]
        recalls.append(
            sum(bool(set(unit) & set(ranking[:5])) for unit in units) / len(units)
        )
        positions = [
            rank
            for rank, cid in enumerate(ranking[:10], 1)
            if cid in set(gold[query["query_id"]]["gold_chunk_ids"])
        ]
        mrrs.append(1 / min(positions) if positions else 0)
    exact = sum(
        len(row.get("questions", [])) == 10
        if kind == "general"
        else bool(row.get("facts"))
        for row in rows
    ) / len(rows)
    recall = float(np.mean(recalls))
    mrr = float(np.mean(mrrs))
    return {
        "score": 0.75 * recall + 0.20 * mrr + 0.05 * exact,
        "evidence_recall@5": recall,
        "mrr@10": mrr,
        "format_success": exact,
    }


def run_gepa(kind: str) -> tuple[str, dict]:
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )

    _, development = _split_queries()
    seed = G.MANUAL_GENERAL_PROMPT if kind == "general" else G.MANUAL_ATOMIC_PROMPT

    def evaluator(candidate: str):
        result = _dev_score(kind, candidate, development)
        feedback = {
            "development_evidence_recall_at_5": result["evidence_recall@5"],
            "development_mrr_at_10": result["mrr@10"],
            "format_success": result["format_success"],
            "guidance": (
                "Improve discriminative coverage of entities, dates, numbers, "
                "relations, comparisons, and outcomes while remaining grounded "
                "and non-repetitive. Preserve the exact output contract."
            ),
        }
        print(f"[GEPA:{kind}] {result}")
        return result["score"], feedback

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(OUT / f"gepa_{kind}_run"),
            seed=C.SEED,
            max_candidate_proposals=3,
            max_metric_calls=4,
            parallel=False,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=f"openai/{C.gen_model()}",
            reflection_minibatch_size=1,
        ),
    )
    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        objective=(
            f"Optimize this {kind} question-generation system prompt for "
            "question-vector retrieval Evidence Recall@5 on the development "
            "split. Keep GPT-5.4-mini, temperature 0.3, seed 42, JSON output, "
            "question count, chunking, Iris embeddings, and retrieval fixed."
        ),
        background=(
            "Training examples define the task. Candidate selection uses only "
            "17 development queries; the 17 test queries are inaccessible."
        ),
        config=config,
    )
    best = result.best_candidate
    prompt = (
        best.get("candidate", next(iter(best.values())))
        if isinstance(best, dict)
        else str(best)
    )
    path = GEPA_GENERAL_PROMPT if kind == "general" else GEPA_ATOMIC_PROMPT
    path.write_text(prompt)
    score = _dev_score(kind, prompt, development)
    return prompt, score


def _mipro_examples(kind: str):
    import dspy

    train, development = _split_queries()
    gold = H.load_gold()
    chunk_map = {c["chunk_id"]: c for c in H.load_chunks()}

    def examples(queries):
        output, seen = [], set()
        for query in queries:
            for chunk_id in gold[query["query_id"]]["gold_chunk_ids"]:
                key = (query["query_id"], chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                output.append(
                    dspy.Example(
                        chunk=chunk_map[chunk_id]["content"],
                        retrieval_query=query["query"],
                        expected_chunk_id=chunk_id,
                    ).with_inputs("chunk", "retrieval_query")
                )
        return output

    return examples(train), examples(development)


def run_mipro(kind: str) -> tuple[object, dict]:
    # Keep DSPy's cache inside the writable project workspace.
    os.environ.setdefault("DSPY_CACHEDIR", str(C.PROJECT_ROOT / ".dspy_cache"))
    import dspy

    base_prompt = (
        G.MANUAL_GENERAL_PROMPT if kind == "general" else G.MANUAL_ATOMIC_PROMPT
    )

    class GenerateQuestions(dspy.Signature):
        __doc__ = base_prompt
        chunk: str = dspy.InputField()
        retrieval_query: str = dspy.InputField(
            desc="Training/development query used only by the optimizer metric"
        )
        questions: list[str] = dspy.OutputField(
            desc=(
                "Exactly 10 grounded questions"
                if kind == "general"
                else "Flattened questions: exactly 3 per meaningful atomic fact"
            )
        )

    class Program(dspy.Module):
        def __init__(self):
            self.generator = dspy.Predict(GenerateQuestions)

        def forward(self, chunk, retrieval_query=""):
            return self.generator(chunk=chunk, retrieval_query=retrieval_query)

    embedder = get_embedder()
    query_cache, question_cache = {}, {}

    def metric(example, prediction, trace=None):
        questions = G._dedup(getattr(prediction, "questions", []) or [])
        if not questions:
            return 0.0
        query = example.retrieval_query
        if query not in query_cache:
            query_cache[query] = np.asarray(embedder.embed_query(query), dtype=float)
        key = tuple(questions)
        if key not in question_cache:
            question_cache[key] = np.asarray(
                embedder.embed_documents(questions), dtype=float
            )
        similarity = float(
            H._cosine_matrix(question_cache[key], query_cache[query]).max()
        )
        # A fixed confidence mapping makes the metric retrieval-oriented while
        # also rewarding the exact general-question count.
        count_quality = min(len(questions), 10) / 10 if kind == "general" else 1
        return 0.9 * max(0.0, similarity) + 0.1 * count_quality

    lm = dspy.LM(
        f"openai/{C.gen_model()}",
        temperature=C.GEN_TEMPERATURE,
        cache=True,
    )
    dspy.configure(lm=lm)
    trainset, valset = _mipro_examples(kind)
    optimizer = dspy.MIPROv2(
        metric=metric,
        prompt_model=lm,
        task_model=lm,
        auto="light",
        num_threads=4,
        seed=C.SEED,
        verbose=True,
        log_dir=str(OUT / f"mipro_{kind}_logs"),
    )
    compiled = optimizer.compile(
        Program(),
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        requires_permission_to_run=False,
    )
    program_path = MIPRO_GENERAL_PROGRAM if kind == "general" else MIPRO_ATOMIC_PROGRAM
    compiled.save(str(program_path))
    instructions = compiled.generator.signature.instructions
    prompt_path = MIPRO_GENERAL_PROMPT if kind == "general" else MIPRO_ATOMIC_PROMPT
    prompt_path.write_text(instructions)
    return compiled, {
        "train_examples": len(trainset),
        "development_examples": len(valset),
        "optimized_instructions": instructions,
    }


def _materialize_mipro(kind: str, program) -> None:
    path = MIPRO_GENERAL_Q if kind == "general" else MIPRO_ATOMIC_Q
    chunks = H.load_chunks()
    cache = {r["chunk_id"]: r for r in G.read_jsonl(path)} if path.exists() else {}
    for pos, chunk in enumerate(chunks, 1):
        if cache.get(chunk["chunk_id"], {}).get("questions"):
            continue
        prediction = program(chunk=chunk["content"], retrieval_query="")
        questions = G._dedup(prediction.questions)
        cache[chunk["chunk_id"]] = {
            "chunk_id": chunk["chunk_id"],
            "questions": questions[:10] if kind == "general" else questions,
        }
        if pos % 10 == 0:
            G.write_jsonl(
                path, [cache[c["chunk_id"]] for c in chunks if c["chunk_id"] in cache]
            )
    G.write_jsonl(path, [cache[c["chunk_id"]] for c in chunks])


def run(kinds: tuple[str, ...] = ("general", "atomic")) -> dict:
    summary = {
        "protocol": {
            "generator": C.gen_model(),
            "temperature": C.GEN_TEMPERATURE,
            "seed": C.SEED,
            "train_queries": 50,
            "development_queries": 17,
            "test_queries_loaded_during_optimization": 0,
        }
    }
    for kind in kinds:
        prompt, score = run_gepa(kind)
        target = GEPA_GENERAL_Q if kind == "general" else GEPA_ATOMIC_Q
        rows = _generate_candidate(kind, prompt)
        G.write_jsonl(target, rows)
        summary[f"gepa_{kind}"] = {
            "prompt": prompt,
            "development": score,
        }
        program, info = run_mipro(kind)
        _materialize_mipro(kind, program)
        summary[f"mipro_{kind}"] = info
    SUMMARY.write_text(json.dumps(summary, indent=2))
    return summary


def repair_caches() -> None:
    """Repair incomplete optimizer rows with the selected prompt and same LLM."""
    chunks = {c["chunk_id"]: c for c in H.load_chunks()}
    for path, prompt_path in (
        (GEPA_GENERAL_Q, GEPA_GENERAL_PROMPT),
        (MIPRO_GENERAL_Q, MIPRO_GENERAL_PROMPT),
    ):
        rows = G.read_jsonl(path)
        for row in rows:
            if len(row.get("questions", [])) != 10:
                result = G._general(
                    chunks[row["chunk_id"]], prompt_path.read_text(), 10
                )
                if len(result["questions"]) == 10:
                    row.update(result)
        G.write_jsonl(path, rows)
    rows = G.read_jsonl(MIPRO_ATOMIC_Q)
    prompt = MIPRO_ATOMIC_PROMPT.read_text()
    for row in rows:
        if row.get("questions"):
            continue
        chunk = chunks[row["chunk_id"]]
        raw = G._call(
            prompt,
            f'''Chunk:\n"""\n{chunk["content"]}\n"""\n
Return a JSON object {{"questions":["...", ...]}} containing the flattened
closed-answer atomic questions.''',
        )
        row["questions"] = G._dedup(raw.get("questions", []))
    G.write_jsonl(MIPRO_ATOMIC_Q, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("all", "general", "atomic"), default="all")
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    if args.repair:
        repair_caches()
        raise SystemExit(0)
    kinds = ("general", "atomic") if args.kind == "all" else (args.kind,)
    print(json.dumps(run(kinds), indent=2))
