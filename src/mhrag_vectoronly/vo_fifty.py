"""
50-article, PURE dense-vector-search comparison (no BM25 / RRF), 4 arms:

    A  = original chunk vectors
    B  = 10 general questions per chunk
    E1 = 1 question per sentence
    E3 = 3 questions per sentence   (E1/E3 derived from one 3-per-sentence generation)

Isolated namespace ``mhrag_vo50`` so the 15-article experiment is untouched. Reuses
vo_data helpers (selection/cleaning/tokenizer/gold), vo_three_arms generation
prompts, vo_retrieval (dense math) and vo_metrics (metric suite). gpt-5.4-mini
generator, Octen-0.6B embedder, cosine. Writes report/mhrag_50articles_vectoronly.html.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make src/ importable

import numpy as np
import chromadb

import config as base
import vo_retrieval as VR
import vo_metrics as VM
import vo_three_arms as T  # reuse generation prompt functions
from embeddings import get_embedder, embedding_signature
from vo_data import (
    _aid,
    _short_type,
    _query_article_ids,
    clean_body,
    _locate_fact,
    read_jsonl,
    _write_jsonl,
    load_corpus,
    load_all_queries,
    _tok_words,
    _ENC,
)

# ---- config ---- #
SEED = 42
ARTICLE_COUNT = 50
CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK = 256, 50, 80
RANK_DEPTH = 10
CAND_MULT = 10
KS = [1, 3, 4, 5, 10]
VM.KS = KS

NS = "mhrag_vo50"
DATA_DIR = base.PROCESSED_DIR / NS
CHROMA_DIR = base.CHROMA_PERSIST_DIR / NS
REPORT = base.PROJECT_ROOT / "report" / "mhrag_50articles_vectoronly.html"
for d in (DATA_DIR, CHROMA_DIR, REPORT.parent):
    d.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
ELIGIBLE = DATA_DIR / "eligible_queries.jsonl"
GOLD = DATA_DIR / "gold_chunk_mapping.jsonl"
GEN_PATH = DATA_DIR / "general_questions.jsonl"
SENT_PATH = DATA_DIR / "sentence3_questions.jsonl"

BASE_COLL, GEN_COLL, SENT1_COLL, SENT3_COLL = (
    "vo50_chunks",
    "vo50_general",
    "vo50_sent1",
    "vo50_sent3",
)


def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def reset_collection(name):
    c = _client()
    try:
        c.delete_collection(name)
    except Exception:
        pass
    return c.create_collection(name, metadata={"hnsw:space": "cosine"})


def get_collection(name):
    return _client().get_collection(name)


# --------------------------------------------------------------------------- #
# Data: select 50 -> clean -> token-chunk -> gold-align
# --------------------------------------------------------------------------- #
def token_chunk(article_key, rec, cleaned):
    toks = _ENC.encode(cleaned)
    stride = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    windows, i, n = [], 0, len(toks)
    while i < n:
        j = min(i + CHUNK_SIZE, n)
        windows.append((i, j))
        if j >= n:
            break
        i += stride
    if len(windows) >= 2 and (windows[-1][1] - windows[-1][0]) < MIN_CHUNK:
        windows[-2] = (windows[-2][0], windows[-1][1])
        windows.pop()
    out = []
    for ci, (s, e) in enumerate(windows):
        out.append(
            {
                "chunk_id": f"{article_key}::c{ci}",
                "parent_document_id": _aid(rec),
                "title": rec.get("title", ""),
                "source": rec.get("source", ""),
                "category": rec.get("category", ""),
                "published_at": rec.get("published_at", ""),
                "n_tokens": e - s,
                "text": _ENC.decode(toks[s:e]).strip(),
            }
        )
    return out


def build_data(force=False):
    if not force and CHUNKS_PATH.exists() and ELIGIBLE.exists() and GOLD.exists():
        return {
            "cached": True,
            "chunks": len(read_jsonl(CHUNKS_PATH)),
            "queries": len(read_jsonl(ELIGIBLE)),
        }
    corpus = load_corpus()
    queries = load_all_queries()
    by_id = {_aid(c): c for c in corpus}
    corpus_ids = set(by_id)
    nonnull = [
        q
        for q in queries
        if q["question_type"] != "null_query"
        and q.get("evidence_list")
        and _query_article_ids(q) <= corpus_ids
    ]
    order = nonnull[:]
    random.Random(SEED).shuffle(order)
    selected = set()
    for q in order:
        arts = _query_article_ids(q)
        if len(selected | arts) <= ARTICLE_COUNT:
            selected |= arts
        if len(selected) >= ARTICLE_COUNT:
            break
    ordered = sorted(selected)
    key_of = {a: f"a{i:03d}" for i, a in enumerate(ordered)}
    chunks = []
    for aid in ordered:
        cleaned, _ = clean_body(by_id[aid].get("body", ""))
        chunks.extend(token_chunk(key_of[aid], by_id[aid], cleaned))
    by_art = defaultdict(list)
    for c in chunks:
        by_art[c["parent_document_id"]].append(c)

    eligible, gold_rows, align = [], [], Counter()
    for q in nonnull:
        req = _query_article_ids(q)
        if not (req <= selected):
            continue
        f2c, unres = [], False
        for k, e in enumerate(q["evidence_list"]):
            hits, _s, how = _locate_fact(e.get("fact", ""), by_art.get(e["url"], []))
            align[how] += 1
            if not hits:
                unres = True
            f2c.append({"chunk_ids": hits})
        if unres:
            continue
        gold_ids = sorted({cid for f in f2c for cid in f["chunk_ids"]})
        units = [sorted(set(f["chunk_ids"])) for f in f2c]
        eligible.append(
            {
                "query_id": q["query_id"],
                "query": q["query"].strip(),
                "question_type": _short_type(q["question_type"]),
                "gold_answer": q.get("answer", ""),
                "required_article_ids": sorted(req),
                "n_required_documents": len(req),
                "n_required_evidence_facts": len(f2c),
                "gold_chunk_ids": gold_ids,
                "evidence_units": units,
            }
        )
        gold_rows.append(
            {
                "query_id": q["query_id"],
                "gold_chunk_ids": gold_ids,
                "evidence_units": units,
            }
        )

    _write_jsonl(CHUNKS_PATH, chunks)
    _write_jsonl(ELIGIBLE, eligible)
    _write_jsonl(GOLD, gold_rows)
    print(
        f"[vo50] {len(selected)} articles -> {len(chunks)} chunks, {len(eligible)} queries "
        f"(match {dict(align)})"
    )
    return {"articles": len(selected), "chunks": len(chunks), "queries": len(eligible)}


def load_chunks():
    return read_jsonl(CHUNKS_PATH)


def load_queries():
    return read_jsonl(ELIGIBLE)


def load_gold():
    return {g["query_id"]: g for g in read_jsonl(GOLD)}


# --------------------------------------------------------------------------- #
# Generation (reuse vo_three_arms prompts) — general 10Q + 3-per-sentence
# --------------------------------------------------------------------------- #
def generate(force=False):
    chunks = load_chunks()
    gcache = (
        {r["chunk_id"]: r for r in read_jsonl(GEN_PATH)}
        if (not force and GEN_PATH.exists())
        else {}
    )
    scache = (
        {r["chunk_id"]: r for r in read_jsonl(SENT_PATH)}
        if (not force and SENT_PATH.exists())
        else {}
    )
    todo_g = [c for c in chunks if c["chunk_id"] not in gcache]
    todo_s = [c for c in chunks if c["chunk_id"] not in scache]
    print(f"[vo50] general to-do {len(todo_g)} | sentence to-do {len(todo_s)}")
    client = base.get_openai_client()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        fg = {ex.submit(T._gen_general, client, c): c for c in todo_g}
        fs = {ex.submit(T._gen_sentence, client, c): c for c in todo_s}
        for fut in as_completed(list(fg)):
            c = fg[fut]
            gcache[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "questions": fut.result(),
            }
        for fut in as_completed(list(fs)):
            c = fs[fut]
            scache[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "sentences": fut.result(),
            }
    _write_jsonl(
        GEN_PATH, [gcache[c["chunk_id"]] for c in chunks if c["chunk_id"] in gcache]
    )
    _write_jsonl(
        SENT_PATH, [scache[c["chunk_id"]] for c in chunks if c["chunk_id"] in scache]
    )
    print(f"[vo50] generation done in {time.perf_counter() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def _index(collection, items):
    embedder = get_embedder()
    chunks = {c["chunk_id"]: c for c in load_chunks()}
    coll = reset_collection(collection)
    vecs = embedder.embed_documents([q for q, _ in items])
    ids, docs, metas, cnt = [], [], [], {}
    for (q, cid), v in zip(items, vecs):
        j = cnt.get(cid, 0)
        cnt[cid] = j + 1
        ids.append(f"{cid}::v{j}")
        docs.append(q)
        metas.append(
            {
                "generated_question_id": f"{cid}::v{j}",
                "parent_chunk_id": cid,
                "parent_document_id": chunks[cid]["parent_document_id"],
            }
        )
    for i in range(0, len(ids), 2000):
        coll.add(
            ids=ids[i : i + 2000],
            embeddings=vecs[i : i + 2000],
            documents=docs[i : i + 2000],
            metadatas=metas[i : i + 2000],
        )
    return len(ids)


def build_indexes():
    chunks = load_chunks()
    gen = {r["chunk_id"]: r for r in read_jsonl(GEN_PATH)}
    sent = {r["chunk_id"]: r for r in read_jsonl(SENT_PATH)}
    # baseline chunk vectors
    embedder = get_embedder()
    coll = reset_collection(BASE_COLL)
    vecs = embedder.embed_documents([c["text"] for c in chunks])
    coll.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vecs,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "parent_chunk_id": c["chunk_id"],
                "parent_document_id": c["parent_document_id"],
            }
            for c in chunks
        ],
    )
    nb = len(chunks)
    ng = _index(
        GEN_COLL,
        [(q, c["chunk_id"]) for c in chunks for q in gen[c["chunk_id"]]["questions"]],
    )
    ns1 = _index(
        SENT1_COLL,
        [
            (s["questions"][0], c["chunk_id"])
            for c in chunks
            for s in sent[c["chunk_id"]]["sentences"]
            if s["questions"]
        ],
    )
    ns3 = _index(
        SENT3_COLL,
        [
            (q, c["chunk_id"])
            for c in chunks
            for s in sent[c["chunk_id"]]["sentences"]
            for q in s["questions"][:3]
        ],
    )
    print(f"[vo50] vectors A={nb} B={ng} E1={ns1} E3={ns3}")
    return {"A": nb, "B": ng, "E1": ns1, "E3": ns3}


# --------------------------------------------------------------------------- #
# Evaluate (pure dense) + render
# --------------------------------------------------------------------------- #
def _ci(x, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _dci(base_v, arm_v, n=1000, seed=42):
    d = arm_v - base_v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    return d.mean(), lo, hi, (lo > 0 or hi < 0)


def evaluate(vecs):
    queries = load_queries()
    gold = load_gold()
    embedder = get_embedder()
    ca = get_collection(BASE_COLL)
    cb = get_collection(GEN_COLL)
    ce1 = get_collection(SENT1_COLL)
    ce3 = get_collection(SENT3_COLL)
    R = {k: [] for k in ("A", "B", "E1", "E3")}
    colls = {"B": cb, "E1": ce1, "E3": ce3}
    for q in queries:
        g = gold[q["query_id"]]
        qv = embedder.embed_query(q["query"])
        common = {
            "query_id": q["query_id"],
            "question_type": q["question_type"],
            "n_required_documents": q["n_required_documents"],
            "n_required_evidence_facts": q["n_required_evidence_facts"],
            "gold_chunk_ids": g["gold_chunk_ids"],
            "evidence_units": g["evidence_units"],
            "required_article_ids": q["required_article_ids"],
        }
        R["A"].append({**common, "ranked": VR.retrieve_baseline(ca, qv, RANK_DEPTH)})
        for tag, coll in colls.items():
            R[tag].append(
                {
                    **common,
                    "ranked": VR.retrieve_generated(coll, qv, RANK_DEPTH, CAND_MULT),
                }
            )
    PQ = {t: {r["query_id"]: VM.per_query(r) for r in rows} for t, rows in R.items()}
    qids = sorted(PQ["A"])

    def vals(P, key):
        return np.array([P[q][key] for q in qids])

    METS = [
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ]
    arms = [
        ("A", "Original chunks", PQ["A"], vecs["A"]),
        ("B", "General 10 questions / chunk", PQ["B"], vecs["B"]),
        ("E1", "1 question per sentence", PQ["E1"], vecs["E1"]),
        ("E3", "3 questions per sentence", PQ["E3"], vecs["E3"]),
    ]
    cd = {}
    for mk, _ in METS:
        b = vals(PQ["A"], mk)
        cd[mk] = {}
        for tag, _l, P, _v in arms:
            v = vals(P, mk)
            lo, hi = _ci(v)
            if tag == "A":
                cd[mk][tag] = dict(mean=v.mean(), lo=lo, hi=hi, d=None)
            else:
                dm, dlo, dhi, sig = _dci(b, v)
                cd[mk][tag] = dict(
                    mean=v.mean(), lo=lo, hi=hi, d=dm, dlo=dlo, dhi=dhi, sig=sig
                )

    def wtl(P):
        d = vals(P, "evidence_recall@5") - vals(PQ["A"], "evidence_recall@5")
        return (
            int((d > 1e-9).sum()),
            int((abs(d) <= 1e-9).sum()),
            int((d < -1e-9).sum()),
        )

    wtls = {t: wtl(P) for t, _l, P, _v in arms if t != "A"}
    _render(arms, cd, METS, wtls, len(qids))
    for mk, lbl in METS:
        print(
            f"  {lbl:<22} "
            + "  ".join(f"{t}={cd[mk][t]['mean']:.3f}" for t, _l, _P, _v in arms)
        )


def _render(arms, cd, METS, wtls, n):
    def cell(mk, tag):
        o = cd[mk][tag]
        if tag == "A":
            return f'<div class="v">{o["mean"]:.3f}</div><div class="ci">[{o["lo"]:.3f}, {o["hi"]:.3f}]</div>'
        cls = "good" if o["d"] > 1e-9 else ("bad" if o["d"] < -1e-9 else "flat")
        star = "<b>*</b>" if o["sig"] else ""
        return (
            f'<div class="v">{o["mean"]:.3f}</div>'
            f'<div class="d {cls}" title="Δ 95% CI [{o["dlo"]:+.3f}, {o["dhi"]:+.3f}]">({o["d"]:+.3f}{star})</div>'
        )

    rows = ""
    for tag, name, _P, v in arms:
        tds = "".join(f"<td>{cell(mk, tag)}</td>" for mk, _ in METS)
        vc = "" if tag == "A" else f'<div class="d flat">(×{v / arms[0][3]:.1f})</div>'
        rows += f'<tr class="{"base" if tag == "A" else ""}"><td class="nm"><b>{tag}</b> · {escape(name)}</td>{tds}<td>{v}{vc}</td></tr>'
    head = "".join(f"<th>{lbl}</th>" for _, lbl in METS)
    wtl_line = "Paired per-query, Evidence Recall@5 vs A — " + " &nbsp;·&nbsp; ".join(
        f"<b>{t}:</b> {wtls[t][0]} W / {wtls[t][1]} T / {wtls[t][2]} L"
        for t, _l, _P, _v in arms
        if t != "A"
    )
    html = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>50-article MultiHop-RAG — pure vector search</title><style>
:root{{--bg:#fff;--fg:#161a21;--muted:#5b6572;--line:#e3e7ee;--card:#f7f9fc;--good:#0a7d3f;--bad:#c0392b;--flat:#8792a1;--hl:#fff4d1;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#242b36;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#8792a1;--hl:#39310c;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding:34px 22px 70px}}h1{{font-size:22px;margin:0 0 4px}}.cap{{color:var(--muted);font-size:12.5px;margin:6px 0 14px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}th,td{{border:1px solid var(--line);padding:8px;text-align:center;vertical-align:middle}}
th{{background:var(--card);font-weight:600}}td.nm,th:first-child{{text-align:left}}tr.base{{background:var(--card)}}tr.base td.nm{{font-weight:700}}
.v{{font-weight:700;font-size:14px}}.ci{{font-size:10px;color:var(--muted)}}.d{{font-size:11px;margin-top:1px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}
.wtl{{font-size:12.5px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:6px 0}}.leg{{font-size:12px;color:var(--muted);margin-top:8px}}
</style></head><body><div class="wrap">
<h1>50-article MultiHop-RAG — pure dense vector search (no BM25 / RRF)</h1>
<p class="cap">A = chunks · B = 10 general Q/chunk · E1 = 1 Q/sentence · E3 = 3 Q/sentence. n={n} queries · Octen-0.6B (dim 1024, cosine) · gpt-5.4-mini. Each non-baseline cell: value (Δ vs A); green/red/gray; <b>*</b> = delta 95% bootstrap CI (1000 resamples) excludes 0. Vectors = index cost (×A).</p>
<table><thead><tr><th>Condition</th>{head}<th>Vectors</th></tr></thead><tbody>{rows}</tbody></table>
<div class="wtl">{wtl_line}</div>
<p class="leg">Closed 50-article pilot — not full-corpus MultiHop-RAG. Pure vector search; no BM25/hybrid/rerank. Isolated namespace mhrag_vo50.</p>
</div></body></html>"""
    REPORT.write_text(html, encoding="utf-8")
    print(f"[vo50] wrote {REPORT}")


def run():
    build_data()
    generate()
    vecs = build_indexes()
    evaluate(vecs)


if __name__ == "__main__":
    run()
