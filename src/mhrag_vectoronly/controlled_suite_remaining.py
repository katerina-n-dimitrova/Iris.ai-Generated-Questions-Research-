"""QueStER-style Qwen rewriting and sparse atomic retrieval conditions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import controlled_suite_generate as G
import controlled_suite_retrieval as R
import vo_config as C
import vo_hierarchical_hybrid as H

OUT = R.RESULTS
QUESTER_CACHE = OUT / "quester_qwen_keyword_queries.jsonl"
REMAINING_METRICS = OUT / "metrics_remaining_conditions.json"
QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
QUESTER_PROMPT = """Rewrite the user question as a compact BM25 keyword search
specification. Preserve essential names, organizations, dates, numbers,
comparisons, negations, and quoted terminology. Remove conversational filler.
Do not answer the question. Return JSON only:
{"keywords":"space-separated keyword query"}"""


def _test_queries() -> list[dict]:
    split = {row["query_id"]: row["split"] for row in G.read_jsonl(R.SPLIT_PATH)}
    return [q for q in H.load_queries() if split[q["query_id"]] == "test"]


def run_quester() -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    queries = _test_queries()
    cache = (
        {row["query_id"]: row for row in G.read_jsonl(QUESTER_CACHE)}
        if QUESTER_CACHE.exists()
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, local_files_only=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, torch_dtype="auto", local_files_only=True
    ).to(device)
    model.eval()
    for pos, query in enumerate(queries, 1):
        if cache.get(query["query_id"], {}).get("keywords"):
            continue
        messages = [
            {"role": "system", "content": QUESTER_PROMPT},
            {"role": "user", "content": query["query"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        answer = tokenizer.decode(
            generated[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        try:
            keywords = json.loads(answer)["keywords"]
        except Exception:
            keywords = re.sub(r"[{}\"']", " ", answer).strip()
        cache[query["query_id"]] = {
            "query_id": query["query_id"],
            "original_query": query["query"],
            "keywords": str(keywords).strip(),
            "model": QWEN_MODEL,
            "prompt_version": "quester_keyword_v1",
            "temperature": 0,
        }
        G.write_jsonl(
            QUESTER_CACHE,
            [cache[q["query_id"]] for q in queries if q["query_id"] in cache],
        )
        print(f"[QueStER] {pos}/{len(queries)}")

    chunks = H.load_chunks()
    gold = H.load_gold()
    chunk_map = {c["chunk_id"]: c for c in chunks}
    rows = []
    for query in queries:
        ranking = R._bm25(chunks, cache[query["query_id"]]["keywords"])
        rows.append(R._metric_row(query, gold[query["query_id"]], ranking, chunk_map))
    return {
        "experiment": "Experiment 1",
        "condition": "E1-QueStER",
        "config": "general512",
        "retrieval_method": "quester_bm25",
        "test_queries": 17,
        "stored_vectors": 0,
        "fallback_count": None,
        "fallback_rate": None,
        "route_analysis": None,
        "metrics": R._evaluate(rows),
        "question_generation": "None; query-side keyword rewriting",
        "prompt_version": "quester_keyword_v1",
        "generation_model": QWEN_MODEL,
    }


def _atomic_dense_rankings() -> tuple[list[dict], dict, dict]:
    queries = _test_queries()
    dataset = R._dataset(512, G.MANUAL_ATOMIC_512)
    qvectors = R._query_vectors(H.load_queries())
    chunk_vectors, question_vectors = R._index(
        "atomic512", dataset["chunks"], dataset["questions"]
    )
    routes = {
        q["query_id"]: R._route_rankings(
            dataset, q, qvectors[q["query_id"]], chunk_vectors, question_vectors
        )
        for q in queries
    }
    return queries, dataset, routes


def run_splade() -> dict:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_name = "naver/splade-cocondenser-ensembledistil"
    queries, dataset, routes = _atomic_dense_rankings()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    def encode(texts):
        output = []
        for start in range(0, len(texts), 8):
            inputs = tokenizer(
                texts[start : start + 8],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = model(**inputs).logits
                weights = torch.log1p(torch.relu(logits.float()))
                weights = weights * inputs["attention_mask"].unsqueeze(-1)
                values = weights.max(dim=1).values.cpu().numpy()
                output.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
        return np.vstack(output)

    chunk_sparse = encode([c["content"] for c in dataset["chunks"]])
    query_sparse = encode([q["query"] for q in queries])
    rows = []
    for query, qvec in zip(queries, query_sparse):
        scores = np.einsum(
            "ij,j->i",
            chunk_sparse.astype(np.float64),
            qvec.astype(np.float64),
            dtype=np.float64,
        )
        sparse = [dataset["chunks"][i]["chunk_id"] for i in H._rank_indices(scores)]
        ranking = R._rrf([routes[query["query_id"]]["question"], sparse])
        rows.append(
            R._metric_row(
                query,
                dataset["gold"][query["query_id"]],
                ranking,
                dataset["chunk_map"],
            )
        )
    return {
        "experiment": "Experiment 3",
        "condition": "E3-SPLADE",
        "config": "atomic512",
        "retrieval_method": "atomic_splade_rrf",
        "test_queries": 17,
        "stored_vectors": len(dataset["questions"]) + len(dataset["chunks"]),
        "fallback_count": None,
        "fallback_rate": None,
        "route_analysis": None,
        "metrics": R._evaluate(rows),
        "supporting_model": model_name,
    }


def run_bge_m3() -> dict:
    from FlagEmbedding import BGEM3FlagModel

    model_name = "BAAI/bge-m3"
    queries, dataset, routes = _atomic_dense_rankings()
    model = BGEM3FlagModel(model_name, use_fp16=False)
    chunk_lex = model.encode(
        [c["content"] for c in dataset["chunks"]],
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )["lexical_weights"]
    query_lex = model.encode(
        [q["query"] for q in queries],
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )["lexical_weights"]

    def lexical_score(query_weights, document_weights):
        return sum(
            float(weight) * float(document_weights.get(token, 0.0))
            for token, weight in query_weights.items()
        )

    rows = []
    for query, qweights in zip(queries, query_lex):
        scores = [lexical_score(qweights, dweights) for dweights in chunk_lex]
        sparse = [dataset["chunks"][i]["chunk_id"] for i in H._rank_indices(scores)]
        ranking = R._rrf([routes[query["query_id"]]["question"], sparse])
        rows.append(
            R._metric_row(
                query,
                dataset["gold"][query["query_id"]],
                ranking,
                dataset["chunk_map"],
            )
        )
    return {
        "experiment": "Experiment 3",
        "condition": "E3-BGE-M3-Sparse",
        "config": "atomic512",
        "retrieval_method": "atomic_bge_m3_sparse_rrf",
        "test_queries": 17,
        "stored_vectors": len(dataset["questions"]) + len(dataset["chunks"]),
        "fallback_count": None,
        "fallback_rate": None,
        "route_analysis": None,
        "metrics": R._evaluate(rows),
        "supporting_model": model_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("quester", "splade", "bge", "all"), default="all"
    )
    args = parser.parse_args()
    existing = (
        json.loads(REMAINING_METRICS.read_text()).get("conditions", [])
        if REMAINING_METRICS.exists()
        else []
    )
    by_name = {row["condition"]: row for row in existing}
    stages = ("quester", "splade", "bge") if args.stage == "all" else (args.stage,)
    for stage in stages:
        row = {
            "quester": run_quester,
            "splade": run_splade,
            "bge": run_bge_m3,
        }[stage]()
        by_name[row["condition"]] = row
        REMAINING_METRICS.write_text(
            json.dumps({"conditions": list(by_name.values())}, indent=2)
        )
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
