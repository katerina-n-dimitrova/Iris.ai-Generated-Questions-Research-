"""
15-article, 3-arm dense-retrieval analysis on ONE closed collection (gpt-5.4-mini):

    A = original chunk vectors                    (baseline)
    B = 10 general questions per chunk
    E = 2 questions per SENTENCE (sentence-focused / "atomic")

Same 15 articles, 154 chunks, 84 eligible queries, embedding model, and generator.
Reuses vo_data (chunks/gold/queries), vo_retrieval (retrieval math), vo_metrics
(metric suite). Produces the spec-format results table (Evidence Recall@1/5/10,
Full-evidence@5, MRR@10, Vectors — value + Δ vs A, 95% bootstrap CIs, * when the
delta CI excludes 0, W/T/L for Recall@5) plus an example chunk showing both
question sets. Writes report/mhrag_15articles_three_arms.html.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from html import escape
from typing import Dict, List

import numpy as np

import vo_config as C
import vo_data as D
import vo_retrieval as VR
import vo_metrics as VM
from embeddings import get_embedder

VM.KS = C.TOP_K_VALUES
KS = C.TOP_K_VALUES

GEN_COLL = "mhrag_vo15_general_q"
GEN15_COLL = "mhrag_vo15_general15_q"  # first 15 of a 20-question generation
GEN20_COLL = "mhrag_vo15_general20_q"  # all 20
GEN20_PATH = C.DATA_DIR / "three_arms_general20_questions.jsonl"
SENT1_COLL = "mhrag_vo15_sentence1_q"  # 1 question per sentence
SENT2_COLL = "mhrag_vo15_sentence_q"  # 2 questions per sentence
SENT3_COLL = "mhrag_vo15_sentence3_q"  # 3 questions per sentence
GEN_PATH = C.DATA_DIR / "three_arms_general_questions.jsonl"
# One generation of up to 3 questions/sentence; the 1/2/3-per-sentence arms are
# the first 1/2/3 of each sentence's questions (isolates count from generation luck).
SENT_PATH = C.DATA_DIR / "three_arms_sentence3_questions.jsonl"
REPORT = C.PROJECT_ROOT / "report" / "mhrag_15articles_three_arms.html"
_WS = re.compile(r"\s+")


def _norm(t):
    return _WS.sub(" ", (t or "")).strip().lower()


# --------------------------------------------------------------------------- #
# Generation (cached)
# --------------------------------------------------------------------------- #
_GEN_SYS = (
    "You write search questions for a news-article passage. Output ONLY a JSON "
    "object. Every question must be answerable using only this passage."
)
_GEN_USER = """Passage:
\"\"\"
{chunk}
\"\"\"

Write EXACTLY 10 natural-language questions this passage can answer. Cover different
facts (don't paraphrase one question 10 times). Name the central entity in each;
never say "the passage/article/text"; avoid vague questions like "what happened".
Return JSON: {{"questions": ["...", ...]}}"""

_SENT_SYS = (
    "You split a passage into sentences and write focused search questions for "
    "each sentence. Output ONLY a JSON object. Every question must be answerable "
    "from its own sentence."
)
_SENT_USER = """Passage:
\"\"\"
{chunk}
\"\"\"

Split the passage into its sentences. For EACH sentence, write EXACTLY 3 focused
questions that are answerable from that sentence alone, ordered most-important first.
Name the central entity in each question; never say "the passage/article/text"; keep
important names, dates and numbers. Return JSON:
{{"sentences":[{{"sentence":"<verbatim sentence>","questions":["q1","q2","q3"]}}]}}"""


def _dedup(qs: List[str]) -> List[str]:
    out, seen = [], set()
    for q in qs:
        q = (q or "").strip()
        k = _norm(q)
        if (
            q
            and k not in seen
            and not any(SequenceMatcher(None, k, s).ratio() >= 0.95 for s in seen)
        ):
            seen.add(k)
            out.append(q)
    return out


def _gen_general(client, chunk):
    for _ in range(C.GEN_MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=C.gen_model(),
                temperature=C.GEN_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _GEN_SYS},
                    {"role": "user", "content": _GEN_USER.format(chunk=chunk["text"])},
                ],
            )
            qs = _dedup(json.loads(r.choices[0].message.content).get("questions", []))
            if qs:
                return qs[:10]
        except Exception:
            continue
    return []


_GEN20_USER = """Passage:
\"\"\"
{chunk}
\"\"\"

Write EXACTLY 20 natural-language questions this passage can answer, ordered
most-important / most-discriminative FIRST. Cover different facts (don't paraphrase);
name the central entity in each; never say "the passage/article/text"; avoid vague
questions like "what happened". Return JSON: {{"questions": ["...", ...]}}"""


def _gen_general20(client, chunk):
    for _ in range(C.GEN_MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=C.gen_model(),
                temperature=C.GEN_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _GEN_SYS},
                    {
                        "role": "user",
                        "content": _GEN20_USER.format(chunk=chunk["text"]),
                    },
                ],
            )
            qs = _dedup(json.loads(r.choices[0].message.content).get("questions", []))
            if qs:
                return qs[:20]
        except Exception:
            continue
    return []


def _gen_sentence(client, chunk):
    for _ in range(C.GEN_MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=C.gen_model(),
                temperature=C.GEN_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SENT_SYS},
                    {"role": "user", "content": _SENT_USER.format(chunk=chunk["text"])},
                ],
            )
            data = json.loads(r.choices[0].message.content).get("sentences", [])
            sents = []
            for s in data:
                qs = _dedup(s.get("questions", []))[:3]
                if qs:
                    sents.append(
                        {"sentence": (s.get("sentence") or "").strip(), "questions": qs}
                    )
            if sents:
                return sents
        except Exception:
            continue
    return []


def generate(force=False):
    chunks = D.load_chunks()
    gcache = (
        {r["chunk_id"]: r for r in D.read_jsonl(GEN_PATH)}
        if (not force and GEN_PATH.exists())
        else {}
    )
    scache = (
        {r["chunk_id"]: r for r in D.read_jsonl(SENT_PATH)}
        if (not force and SENT_PATH.exists())
        else {}
    )
    g20cache = (
        {r["chunk_id"]: r for r in D.read_jsonl(GEN20_PATH)}
        if (not force and GEN20_PATH.exists())
        else {}
    )
    todo_g = [c for c in chunks if c["chunk_id"] not in gcache]
    todo_s = [c for c in chunks if c["chunk_id"] not in scache]
    todo_g20 = [c for c in chunks if c["chunk_id"] not in g20cache]
    print(
        f"[3arms] general to-do {len(todo_g)} | sentence to-do {len(todo_s)} | general20 to-do {len(todo_g20)}"
    )
    client = C.openai_client()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        fg = {ex.submit(_gen_general, client, c): c for c in todo_g}
        fs = {ex.submit(_gen_sentence, client, c): c for c in todo_s}
        fg20 = {ex.submit(_gen_general20, client, c): c for c in todo_g20}
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
        for fut in as_completed(list(fg20)):
            c = fg20[fut]
            g20cache[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "questions": fut.result(),
            }
    D._write_jsonl(
        GEN_PATH, [gcache[c["chunk_id"]] for c in chunks if c["chunk_id"] in gcache]
    )
    D._write_jsonl(
        SENT_PATH, [scache[c["chunk_id"]] for c in chunks if c["chunk_id"] in scache]
    )
    D._write_jsonl(
        GEN20_PATH,
        [g20cache[c["chunk_id"]] for c in chunks if c["chunk_id"] in g20cache],
    )
    ng = sum(len(gcache[c["chunk_id"]]["questions"]) for c in chunks)
    ns = sum(
        len(q)
        for c in chunks
        for s in scache[c["chunk_id"]]["sentences"]
        for q in [s["questions"]]
    )
    print(
        f"[3arms] general {ng} Q | sentence {ns} Q in {time.perf_counter() - t0:.1f}s"
    )


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def _index(collection, items):
    """items: list of (question, chunk). Returns vector count."""
    embedder = get_embedder()
    chunks = {c["chunk_id"]: c for c in D.load_chunks()}
    coll = C.reset_collection(collection)
    vecs = embedder.embed_documents([q for q, _ in items])
    ids, docs, metas = [], [], []
    counters: Dict[str, int] = {}
    for (q, cid), v in zip(items, vecs):
        j = counters.get(cid, 0)
        counters[cid] = j + 1
        qid = f"{cid}::v{j}"
        ids.append(qid)
        docs.append(q)
        metas.append(
            {
                "generated_question_id": qid,
                "parent_chunk_id": cid,
                "parent_document_id": chunks[cid]["parent_document_id"],
                "title": chunks[cid]["title"],
                "source": chunks[cid]["source"],
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
    gen = {r["chunk_id"]: r for r in D.read_jsonl(GEN_PATH)}
    sent = {r["chunk_id"]: r for r in D.read_jsonl(SENT_PATH)}
    chunks = D.load_chunks()
    # ensure baseline chunk collection exists
    try:
        n = C.get_collection(C.BASELINE_COLLECTION).count()
        if n == 0:
            raise ValueError
    except Exception:
        embedder = get_embedder()
        coll = C.reset_collection(C.BASELINE_COLLECTION)
        vecs = embedder.embed_documents([c["text"] for c in chunks])
        coll.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=vecs,
            documents=[c["text"] for c in chunks],
            metadatas=[
                {
                    "parent_chunk_id": c["chunk_id"],
                    "parent_document_id": c["parent_document_id"],
                    "title": c["title"],
                    "source": c["source"],
                }
                for c in chunks
            ],
        )
    g_items = [
        (q, c["chunk_id"]) for c in chunks for q in gen[c["chunk_id"]]["questions"]
    ]

    def sent_items(k):
        return [
            (q, c["chunk_id"])
            for c in chunks
            for s in sent[c["chunk_id"]]["sentences"]
            for q in s["questions"][:k]
        ]

    gen20 = {r["chunk_id"]: r for r in D.read_jsonl(GEN20_PATH)}

    def gen20_items(k):
        return [
            (q, c["chunk_id"])
            for c in chunks
            for q in gen20[c["chunk_id"]]["questions"][:k]
        ]

    nb = C.get_collection(C.BASELINE_COLLECTION).count()
    ng = _index(GEN_COLL, g_items)
    ng15 = _index(GEN15_COLL, gen20_items(15))
    ng20 = _index(GEN20_COLL, gen20_items(20))
    ns1 = _index(SENT1_COLL, sent_items(1))
    ns2 = _index(SENT2_COLL, sent_items(2))
    ns3 = _index(SENT3_COLL, sent_items(3))
    print(
        f"[3arms] vectors: A={nb} B={ng} B15={ng15} B20={ng20} "
        f"E1={ns1} E2={ns2} E3={ns3}"
    )
    return {"A": nb, "B": ng, "B15": ng15, "B20": ng20, "E1": ns1, "E2": ns2, "E3": ns3}


# --------------------------------------------------------------------------- #
# Evaluate + report
# --------------------------------------------------------------------------- #
def _ci(x, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _dci(base, arm, n=1000, seed=42):
    d = arm - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    return d.mean(), lo, hi, (lo > 0 or hi < 0)


def evaluate(vecs):
    queries = D.load_eligible_queries()
    gold = D.load_gold()
    embedder = get_embedder()
    ca = C.get_collection(C.BASELINE_COLLECTION)
    cb = C.get_collection(GEN_COLL)
    cb15 = C.get_collection(GEN15_COLL)
    cb20 = C.get_collection(GEN20_COLL)
    ce1 = C.get_collection(SENT1_COLL)
    ce2 = C.get_collection(SENT2_COLL)
    ce3 = C.get_collection(SENT3_COLL)
    R = {k: [] for k in ("A", "B", "B15", "B20", "E1", "E2", "E3")}
    colls = {"B": cb, "B15": cb15, "B20": cb20, "E1": ce1, "E2": ce2, "E3": ce3}
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
        R["A"].append({**common, "ranked": VR.retrieve_baseline(ca, qv, C.RANK_DEPTH)})
        for tag, coll in colls.items():
            R[tag].append(
                {
                    **common,
                    "ranked": VR.retrieve_generated(
                        coll, qv, C.RANK_DEPTH, C.CANDIDATE_MULTIPLIER
                    ),
                }
            )
    PQ = {
        tag: {r["query_id"]: VM.per_query(r) for r in rows} for tag, rows in R.items()
    }
    pa = PQ["A"]
    qids = sorted(pa)

    def vals(Pd, key):
        return np.array([Pd[q][key] for q in qids])

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
        ("B15", "General 15 questions / chunk", PQ["B15"], vecs["B15"]),
        ("B20", "General 20 questions / chunk", PQ["B20"], vecs["B20"]),
        ("E1", "1 question per sentence", PQ["E1"], vecs["E1"]),
        ("E2", "2 questions per sentence", PQ["E2"], vecs["E2"]),
        ("E3", "3 questions per sentence", PQ["E3"], vecs["E3"]),
    ]
    celldata = {}
    for mk, _ in METS:
        base = vals(pa, mk)
        celldata[mk] = {}
        for tag, _, P, _v in arms:
            v = vals(P, mk)
            mean = v.mean()
            lo, hi = _ci(v)
            if tag == "A":
                celldata[mk][tag] = dict(mean=mean, lo=lo, hi=hi, d=None)
            else:
                dm, dlo, dhi, sig = _dci(base, v)
                celldata[mk][tag] = dict(
                    mean=mean, lo=lo, hi=hi, d=dm, dlo=dlo, dhi=dhi, sig=sig
                )

    def wtl(P):
        d = vals(P, "evidence_recall@5") - vals(pa, "evidence_recall@5")
        return (
            int((d > 1e-9).sum()),
            int((abs(d) <= 1e-9).sum()),
            int((d < -1e-9).sum()),
        )

    wtls = {tag: wtl(P) for tag, _, P, _v in arms if tag != "A"}
    return METS, arms, celldata, wtls, len(qids)


def render(METS, arms, cd, wtls, n):
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
    tbl = f"<table><thead><tr><th>Condition</th>{head}<th>Vectors</th></tr></thead><tbody>{rows}</tbody></table>"

    # example chunk
    gen = {r["chunk_id"]: r for r in D.read_jsonl(GEN_PATH)}
    sent = {r["chunk_id"]: r for r in D.read_jsonl(SENT_PATH)}
    chunks = {c["chunk_id"]: c for c in D.load_chunks()}
    pick = next(
        (
            cid
            for cid, c in chunks.items()
            if c["category"] == "technology"
            and 220 <= c["n_tokens"] <= 256
            and len(gen[cid]["questions"]) == 10
            and len(sent[cid]["sentences"]) >= 3
        ),
        None,
    )
    if not pick:
        pick = next(cid for cid in chunks if len(gen[cid]["questions"]) == 10)
    c = chunks[pick]
    gql = "".join(f"<li>{escape(q)}</li>" for q in gen[pick]["questions"])
    sents = sent[pick]["sentences"]
    n_sent_q = sum(len(s["questions"]) for s in sents)
    sblocks = "".join(
        f'<div class="sent"><div class="stext">“{escape(s["sentence"])}”</div>'
        f'<ol class="qs">{"".join(f"<li>{escape(q)}</li>" for q in s["questions"])}</ol></div>'
        for s in sents
    )
    example = f"""<h2>Example — one chunk, two ways of forming questions</h2>
<div class="ex"><div class="exhead"><b>Chunk {escape(pick)}</b> · {escape(c["title"])} ({escape(c["source"])}) · {c["n_tokens"]} tokens</div>
<div class="chunk">{escape(c["text"])}</div>
<div class="two">
  <div><div class="qh">B · 10 general questions for the whole chunk</div><ol class="qs">{gql}</ol></div>
  <div><div class="qh">E · sentence-focused — {len(sents)} sentences → {n_sent_q} questions (up to 3 each; E1/E2/E3 use the first 1/2/3)</div>{sblocks}</div>
</div></div>"""

    caption = (
        f"n={n} evaluation queries · k∈{{1,5,10}} · corpus = 154 chunks from 15 articles · "
        f"embedder Octen-Embedding-0.6B (dim 1024, cosine) · generator gpt-5.4-mini. "
        f"Each non-baseline cell: value (Δ vs A); green/red/gray = positive/negative/zero; "
        f"<b>*</b> = delta 95% bootstrap CI (1000 resamples) excludes 0. Vectors is the cost column (×A)."
    )
    wtl_line = "Paired per-query, Evidence Recall@5 vs A — " + " &nbsp;·&nbsp; ".join(
        f"<b>{tag}:</b> {wtls[tag][0]} W / {wtls[tag][1]} T / {wtls[tag][2]} L"
        for tag, _n, _P, _v in arms
        if tag != "A"
    )

    doc = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>15-article MultiHop-RAG — chunks vs general vs sentence questions</title><style>
:root{{--bg:#fff;--fg:#161a21;--muted:#5b6572;--line:#e3e7ee;--card:#f7f9fc;--good:#0a7d3f;--bad:#c0392b;--flat:#8792a1;--hl:#fff4d1;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#242b36;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#8792a1;--hl:#39310c;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:34px 22px 70px}}h1{{font-size:23px;margin:0 0 4px}}h2{{font-size:18px;margin:30px 0 8px}}
.cap{{color:var(--muted);font-size:12.5px;margin:6px 0 12px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}th,td{{border:1px solid var(--line);padding:8px;text-align:center;vertical-align:middle}}
th{{background:var(--card);font-weight:600}}td.nm,th:first-child{{text-align:left}}tr.base{{background:var(--card)}}tr.base td.nm{{font-weight:700}}
.v{{font-weight:700;font-size:14px}}.ci{{font-size:10px;color:var(--muted)}}.d{{font-size:11.5px;margin-top:1px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}
.wtl{{font-size:12.5px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:6px 0}}
.ex{{border:1px solid var(--line);border-radius:10px;padding:16px 18px;background:var(--card)}}.exhead{{font-size:13px;color:var(--muted);margin-bottom:8px}}
.chunk{{font-size:13px;line-height:1.6;white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px 14px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:14px}}@media(max-width:760px){{.two{{grid-template-columns:1fr}}}}
.qh{{font-weight:600;font-size:13px;margin-bottom:6px}}.qs{{margin:0 0 6px;padding-left:20px;font-size:12.5px;line-height:1.55}}
.sent{{margin-bottom:10px}}.stext{{font-size:12.5px;color:var(--muted);font-style:italic}}
.leg{{font-size:12px;color:var(--muted);margin-top:8px}}
</style></head><body><div class="wrap">
<h1>15-article MultiHop-RAG — chunks vs general vs sentence questions</h1>
<p class="cap">Does indexing generated questions instead of the original chunks retrieve gold evidence better? Dense cosine, local ChromaDB, gpt-5.4-mini. Rows = baseline A first, then each arm.</p>
{tbl}
<p class="cap">{caption}</p>
<div class="wtl">{wtl_line}</div>
<p class="leg">Only asterisked deltas are statistically distinguishable from zero; unmarked deltas are not described as improvements or regressions. Closed 15-article pilot — not full-corpus MultiHop-RAG.</p>
{example}
</div></body></html>"""
    REPORT.write_text(doc, encoding="utf-8")
    print(f"[3arms] wrote {REPORT}")


def run():
    generate()
    vecs = build_indexes()
    METS, arms, cd, wtls, n = evaluate(vecs)
    render(METS, arms, cd, wtls, n)
    for mk, lbl in METS:
        vals = "  ".join(f"{tag}={cd[mk][tag]['mean']:.3f}" for tag, _n, _P, _v in arms)
        print(f"  {lbl:<22} {vals}")


if __name__ == "__main__":
    run()
