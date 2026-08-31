"""Seven isolated generated-question-type retrieval experiments on the frozen
10-article MultiHop-RAG collection. Dense cosine only; categories never pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
AM = SRC / "mhrag_atomic_mix"
VO = SRC / "mhrag_vectoronly"
for p in (SRC, AM, VO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config as base
import am_config as AC
import am_data as D
import vo_metrics as VM
from embeddings import embedding_signature, get_embedder

CFG_PATH = base.PROJECT_ROOT / "config" / "mhrag_question_types_10.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text())
SEED = int(CFG["random_seed"])
NQ = int(CFG["questions_per_type_per_chunk"])
TYPES = tuple(CFG["question_types"])
KS = list(CFG["retrieval"]["top_k_values"])
DEPTH = int(CFG["retrieval"]["rank_depth"])
MULT = int(CFG["retrieval"]["candidate_multiplier"])
RT_TOPK = int(CFG["filtering"]["roundtrip_parent_top_k"])
DUP = float(CFG["filtering"]["near_duplicate_cosine"])
NBOOT = int(CFG["evaluation"]["bootstrap_resamples"])
NS = CFG["chromadb"]["namespace"]

DATA = base.PROCESSED_DIR / NS
RESULTS = base.RESULTS_DIR / NS
CHROMA = base.CHROMA_PERSIST_DIR / NS
REPORT = base.PROJECT_ROOT / "report" / f"{NS}_results.html"
for p in (DATA, RESULTS, CHROMA, REPORT.parent):
    p.mkdir(parents=True, exist_ok=True)
RAW = DATA / "typed_questions_raw.jsonl"
ACCEPTED = DATA / "typed_questions_accepted.jsonl"
GEN_REPORT = RESULTS / "generation_report.json"
FILTER_REPORT = RESULTS / "filter_report.json"
INDEX_REPORT = RESULTS / "index_report.json"
RANKINGS = RESULTS / "rankings"
RANKINGS.mkdir(exist_ok=True)
METRICS = RESULTS / "metrics.json"
OVERALL_CSV = RESULTS / "overall_comparison.csv"
MATRIX_CSV = RESULTS / "query_type_matrix.csv"

_WS = re.compile(r"\s+")
_VAGUE = re.compile(
    r"\b(this|that) (company|product|device|event|team|report|article)|\b(the passage|the text|it)\b",
    re.I,
)


def norm(s):
    return _WS.sub(" ", (s or "")).strip().lower()


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def finite_embeddings(embedder, texts):
    """Embed documents and retry rare non-finite batch rows individually."""
    vectors = embedder.embed_documents(texts)

    def invalid(v):
        a = np.asarray(v, dtype=np.float32)
        norm = float(np.linalg.norm(a.astype(np.float64)))
        return not np.isfinite(a).all() or not (0.5 <= norm <= 1.5)

    bad = [i for i, v in enumerate(vectors) if invalid(v)]
    for i in bad:
        vectors[i] = embedder.embed_documents([texts[i]])[0]
    still_bad = [i for i, v in enumerate(vectors) if invalid(v)]
    if still_bad:
        raise ValueError(
            f"embedding model returned non-finite vectors at rows {still_bad[:10]}"
        )
    if bad:
        print(f"[embeddings] retried {len(bad)} non-finite batch rows individually")
    return vectors


SYSTEM = """You create grounded search questions from a news passage. Produce distinct
question views by semantic category. Use only the supplied passage. Return JSON only."""

PROMPT = """Article title: {title}
Source: {source}; Published: {published}

Passage:
\"\"\"{text}\"\"\"

For EACH category below, attempt exactly {n} distinct, self-contained search questions.
Never invent a relationship that is absent. If the passage truly cannot support six
distinct questions of a category, return fewer; never pad with paraphrases.

- atomic: one standalone fact only.
- relation: explicit relationship between named entities, people, organizations, objects, or events.
- causal: an explicit cause, reason, motivation, consequence, or outcome.
- temporal: date/time, ordering, before/after, duration, or change over time.
- comparison: explicit similarity, difference, ranking, contrast, or quantitative comparison.
- compositional: combines at least two passage facts and requires both to answer.
- broad: specific whole-passage or main-event question retaining central named entities.

Every item must contain one short verbatim supporting_span copied from the passage and
a short_answer. Avoid vague references such as "this company", "the article", or "it".

JSON shape:
{{"categories": {{"atomic": [{{"question":"...","short_answer":"...","supporting_span":"..."}}],
"relation": [], "causal": [], "temporal": [], "comparison": [],
"compositional": [], "broad": []}}}}
"""


def generate(force=False):
    chunks = D.load_chunks()
    cached = {} if force else {r["chunk_id"]: r for r in read_jsonl(RAW)}
    todo = [c for c in chunks if c["chunk_id"] not in cached]
    client = base.get_openai_client()

    def one(c):
        prompt = PROMPT.format(
            title=c["title"],
            source=c["source"],
            published=c["published_at"],
            text=c["text"],
            n=NQ,
        )
        last = ""
        for attempt in range(1, int(CFG["generation"]["maximum_retries"]) + 1):
            try:
                response = client.chat.completions.create(
                    model=AC.gen_model(),
                    temperature=float(CFG["generation"]["temperature"]),
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                )
                cats = json.loads(response.choices[0].message.content).get(
                    "categories", {}
                )
                if all(isinstance(cats.get(t, []), list) for t in TYPES):
                    return {
                        "chunk_id": c["chunk_id"],
                        "categories": cats,
                        "attempts": attempt,
                        "error": "",
                    }
                last = "missing category arrays"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
        return {
            "chunk_id": c["chunk_id"],
            "categories": {t: [] for t in TYPES},
            "attempts": int(CFG["generation"]["maximum_retries"]),
            "error": last,
        }

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(CFG["generation"]["workers"])) as ex:
        futures = {ex.submit(one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            cached[row["chunk_id"]] = row
            if i % 12 == 0:
                print(f"[generate] {i}/{len(todo)}", flush=True)
            write_jsonl(
                RAW, [cached[c["chunk_id"]] for c in chunks if c["chunk_id"] in cached]
            )
    rows = [cached[c["chunk_id"]] for c in chunks]
    counts = {
        t: sum(min(len(r["categories"].get(t, [])), NQ) for r in rows) for t in TYPES
    }
    report = {
        "model": AC.gen_model(),
        "chunks": len(chunks),
        "requested_per_type_per_chunk": NQ,
        "requested_total": len(chunks) * len(TYPES) * NQ,
        "raw_counts": counts,
        "failed_chunks": sum(bool(r.get("error")) for r in rows),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    GEN_REPORT.write_text(json.dumps(report, indent=2))
    print("[generate]", report)
    return report


def _reset(client, name):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def client():
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA))


def filter_questions(force=False):
    if ACCEPTED.exists() and FILTER_REPORT.exists() and not force:
        return json.loads(FILTER_REPORT.read_text())
    chunks = {c["chunk_id"]: c for c in D.load_chunks()}
    candidates, rejected = [], Counter()
    for row in read_jsonl(RAW):
        c = chunks[row["chunk_id"]]
        ctext = norm(c["text"])
        for typ in TYPES:
            for j, item in enumerate(row["categories"].get(typ, [])[:NQ]):
                q = norm(item.get("question"))
                span = norm(item.get("supporting_span"))
                if not q or not item.get("short_answer") or not span:
                    rejected["structural"] += 1
                    continue
                if span not in ctext:
                    rejected["ungrounded"] += 1
                    continue
                if _VAGUE.search(item["question"]):
                    rejected["not_self_contained"] += 1
                    continue
                candidates.append(
                    {
                        "question_id": f"{c['chunk_id']}::{typ}::{j}",
                        "question_type": typ,
                        "question": item["question"].strip(),
                        "short_answer": item["short_answer"],
                        "supporting_span": item["supporting_span"],
                        "parent_chunk_id": c["chunk_id"],
                        "parent_document_id": c["parent_document_id"],
                        "title": c["title"],
                        "source": c["source"],
                    }
                )

    emb = get_embedder()
    chunks_list = list(chunks.values())
    cvecs = finite_embeddings(emb, [c["text"] for c in chunks_list])
    qvecs = (
        finite_embeddings(emb, [q["question"] for q in candidates])
        if candidates
        else []
    )
    # The local model may return float16. NumPy's float16 dot reduction can
    # overflow even for normalized 1024-d vectors, so use float32 for all
    # filtering similarities and retain that dtype in the temporary cache.
    cmat = np.ascontiguousarray(cvecs, dtype=np.float64)
    cid_order = [c["chunk_id"] for c in chunks_list]
    by = defaultdict(list)
    for q, v in zip(candidates, qvecs):
        v = np.ascontiguousarray(v, dtype=np.float64)
        # Elementwise reduction avoids an Accelerate/BLAS matmul issue observed
        # on this macOS host for these otherwise finite, unit-length vectors.
        sims = np.sum(cmat * v[None, :], axis=1, dtype=np.float64)
        rank = (
            int(
                np.where(np.argsort(-sims) == cid_order.index(q["parent_chunk_id"]))[0][
                    0
                ]
            )
            + 1
        )
        if rank > RT_TOPK:
            rejected["roundtrip"] += 1
            continue
        q["roundtrip_rank"] = rank
        q["_vec"] = v
        by[(q["parent_chunk_id"], q["question_type"])].append(q)
    accepted = []
    for _, rows in by.items():
        kept = []
        for q in sorted(rows, key=lambda x: x["roundtrip_rank"]):
            if any(
                float(
                    np.sum(
                        np.asarray(q["_vec"], dtype=np.float64)
                        * np.asarray(k["_vec"], dtype=np.float64),
                        dtype=np.float64,
                    )
                )
                >= DUP
                for k in kept
            ):
                rejected["near_duplicate"] += 1
            elif len(kept) < NQ:
                kept.append(q)
        accepted.extend(kept)
    for q in accepted:
        q.pop("_vec", None)
    write_jsonl(ACCEPTED, accepted)
    counts = Counter(q["question_type"] for q in accepted)
    coverage = {
        t: len({q["parent_chunk_id"] for q in accepted if q["question_type"] == t})
        for t in TYPES
    }
    report = {
        "accepted_counts": dict(counts),
        "chunk_coverage": coverage,
        "rejected": dict(rejected),
        "roundtrip_parent_top_k": RT_TOPK,
        "near_duplicate_cosine": DUP,
    }
    FILTER_REPORT.write_text(json.dumps(report, indent=2))
    print("[filter]", report)
    return report


def index():
    chunks = D.load_chunks()
    qs = read_jsonl(ACCEPTED)
    emb = get_embedder()
    cc = client()
    stats = {}
    a = _reset(cc, "baseline")
    av = finite_embeddings(emb, [c["text"] for c in chunks])
    a.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=av,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "parent_chunk_id": c["chunk_id"],
                "parent_document_id": c["parent_document_id"],
            }
            for c in chunks
        ],
    )
    stats["baseline"] = len(chunks)
    for typ in TYPES:
        rows = [q for q in qs if q["question_type"] == typ]
        coll = _reset(cc, typ)
        vecs = finite_embeddings(emb, [q["question"] for q in rows]) if rows else []
        if rows:
            coll.add(
                ids=[q["question_id"] for q in rows],
                embeddings=vecs,
                documents=[q["question"] for q in rows],
                metadatas=[
                    {
                        "parent_chunk_id": q["parent_chunk_id"],
                        "parent_document_id": q["parent_document_id"],
                    }
                    for q in rows
                ],
            )
        stats[typ] = len(rows)
    out = {
        "counts": stats,
        "embedding_model": embedding_signature(),
        "vector_dim": emb.dim,
    }
    INDEX_REPORT.write_text(json.dumps(out, indent=2))
    print("[index]", stats)
    return out


def _retrieve(coll, v, is_baseline=False):
    n = coll.count()
    k = min(DEPTH if is_baseline else max(100, DEPTH * MULT), n)
    res = coll.query(
        query_embeddings=[v],
        n_results=k,
        include=["metadatas", "distances", "documents"],
    )
    best = {}
    for m, d, doc in zip(res["metadatas"][0], res["distances"][0], res["documents"][0]):
        cid = m["parent_chunk_id"]
        score = 1.0 - float(d)
        if cid not in best or score > best[cid]["score"]:
            best[cid] = {
                "chunk_id": cid,
                "parent_document_id": m["parent_document_id"],
                "score": score,
                "best_question": None if is_baseline else doc,
            }
    rows = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:DEPTH]
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def retrieve():
    queries = D.load_eligible_queries()
    gold = D.load_gold()
    emb = get_embedder()
    cc = client()
    conditions = ("baseline",) + TYPES
    out = {c: [] for c in conditions}
    for qi, q in enumerate(queries, 1):
        v = finite_embeddings(emb, [q["query"]])[0]
        g = gold[q["query_id"]]
        common = {
            "query_id": q["query_id"],
            "query": q["query"],
            "question_type": q["question_type"],
            "n_required_documents": q["n_required_documents"],
            "n_required_evidence_facts": q["n_required_evidence_facts"],
            "gold_chunk_ids": g["gold_chunk_ids"],
            "evidence_units": g["evidence_units"],
            "required_article_ids": q["required_article_ids"],
        }
        for cond in conditions:
            ranked = _retrieve(cc.get_collection(cond), v, cond == "baseline")
            out[cond].append({**common, "condition": cond, "ranked": ranked})
        if qi % 15 == 0:
            print(f"[retrieve] {qi}/{len(queries)}", flush=True)
    for cond, rows in out.items():
        write_jsonl(RANKINGS / f"{cond}.jsonl", rows)
    return {c: len(v) for c, v in out.items()}


def evaluate():
    VM.KS = KS
    VM.C.SEED = SEED
    VM.C.BOOTSTRAP_RESAMPLES = NBOOT
    conditions = ("baseline",) + TYPES
    per = {
        c: [VM.per_query(r) for r in read_jsonl(RANKINGS / f"{c}.jsonl")]
        for c in conditions
    }
    agg = VM.aggregate(per)
    paired = {t: VM.paired(per["baseline"], per[t]) for t in TYPES}
    query_types = sorted({r["question_type"] for r in per["baseline"]})
    matrix = {}
    for qt in query_types:
        matrix[qt] = {}
        for cond in conditions:
            rows = [r for r in per[cond] if r["question_type"] == qt]
            matrix[qt][cond] = round(
                float(np.mean([r["evidence_recall@5"] for r in rows])), 4
            )
    result = {
        "n_queries": len(per["baseline"]),
        "k_values": KS,
        "overall": agg,
        "paired_vs_baseline": paired,
        "query_type_er5_matrix": matrix,
        "index_counts": json.loads(INDEX_REPORT.read_text())["counts"],
    }
    METRICS.write_text(json.dumps(result, indent=2))
    with OVERALL_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["condition", "vectors"]
            + [f"evidence_recall@{k}" for k in KS]
            + ["delta_er5", "p_er5"]
        )
        base5 = agg["baseline"]["evidence_recall@5"]["mean"]
        for c in conditions:
            w.writerow(
                [c, result["index_counts"][c]]
                + [agg[c][f"evidence_recall@{k}"]["mean"] for k in KS]
                + (
                    [0, ""]
                    if c == "baseline"
                    else [
                        round(agg[c]["evidence_recall@5"]["mean"] - base5, 4),
                        paired[c]["evidence_recall@5"]["paired_p"],
                    ]
                )
            )
    with MATRIX_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark_query_type"] + list(conditions))
        for qt in query_types:
            w.writerow([qt] + [matrix[qt][c] for c in conditions])
    print(
        "[evaluate] ER@5", {c: agg[c]["evidence_recall@5"]["mean"] for c in conditions}
    )
    return result


def report():
    m = json.loads(METRICS.read_text())
    gen = json.loads(GEN_REPORT.read_text())
    fil = json.loads(FILTER_REPORT.read_text())
    conditions = ("baseline",) + TYPES
    base5 = m["overall"]["baseline"]["evidence_recall@5"]["mean"]
    rows = "".join(
        f"<tr><td>{escape(c)}</td><td>{m['index_counts'][c]}</td>"
        + "".join(
            f"<td>{m['overall'][c][f'evidence_recall@{k}']['mean']:.3f}</td>"
            for k in KS
        )
        + (
            "<td>—</td><td>—</td>"
            if c == "baseline"
            else f"<td>{m['overall'][c]['evidence_recall@5']['mean'] - base5:+.3f}</td><td>{m['paired_vs_baseline'][c]['evidence_recall@5']['paired_p']}</td>"
        )
        + "</tr>"
        for c in conditions
    )
    matrix = "".join(
        "<tr><td>"
        + escape(qt)
        + "</td>"
        + "".join(f"<td>{vals[c]:.3f}</td>" for c in conditions)
        + "</tr>"
        for qt, vals in m["query_type_er5_matrix"].items()
    )
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>MultiHop-RAG question-type ablation</title>
<style>body{{font:15px system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#17202a}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #ccd3da;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f2f5f7}}.note{{background:#f6f8fa;padding:14px;border-left:4px solid #476d9b}}</style></head><body>
<h1>MultiHop-RAG: isolated generated-question types</h1><p class=note>Same frozen 10 articles, 96 chunks, {m["n_queries"]} queries, gold labels, embedding model and cosine retrieval. Categories are indexed and evaluated separately; never pooled. Six questions were requested per type/chunk. Model: {escape(gen["model"])}.</p>
<h2>Overall Evidence Recall</h2><table><tr><th>Condition</th><th>Vectors</th>{"".join(f"<th>ER@{k}</th>" for k in KS)}<th>Δ ER@5</th><th>paired p</th></tr>{rows}</table>
<h2>Evidence Recall@5: benchmark query type × indexed representation</h2><table><tr><th>Benchmark type</th>{"".join(f"<th>{escape(c)}</th>" for c in conditions)}</tr>{matrix}</table>
<h2>Generation and filtering</h2><pre>{escape(json.dumps({"raw_counts": gen["raw_counts"], "accepted_counts": fil["accepted_counts"], "chunk_coverage": fil["chunk_coverage"], "rejected": fil["rejected"]}, indent=2))}</pre>
</body></html>"""
    REPORT.write_text(html)
    print(f"[report] {REPORT}")
    return str(REPORT)


STAGES = {
    "generate": generate,
    "filter": filter_questions,
    "index": index,
    "retrieve": retrieve,
    "evaluate": evaluate,
    "report": report,
}
ORDER = ("generate", "filter", "index", "retrieve", "evaluate", "report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("all",) + tuple(STAGES), default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for s in ORDER if a.stage == "all" else (a.stage,):
        print(f"\n===== {s} =====")
        STAGES[s](force=a.force) if s in ("generate", "filter") else STAGES[s]()


if __name__ == "__main__":
    main()
