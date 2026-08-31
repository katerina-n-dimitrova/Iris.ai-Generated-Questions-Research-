"""
15-article MultiHop-RAG — configurable chunk-size/overlap sub-study.

Same closed 15-article pilot as vo_three_arms, re-chunked according to the
values in config/mhrag_vectoronly.yaml. It compares the four established arms
plus a Promptagator-style few-shot arm:

    A  = original chunk vectors                (baseline)
    B  = 10 general questions per chunk
    E1 = 1 question per sentence
    E3 = 3 questions per sentence   (E1/E3 both taken from one 3-per-sentence
                                     generation: E1 = first question of each
                                     sentence, E3 = first three)
    P8 = 8 style-matched questions per chunk, with leakage-free examples
    D30 = Doc2Query++ topic/keyword planning, then 30 questions per chunk

Embedder is whatever EMBEDDING_BACKEND points at in .env (this run: the Iris
hosted service, dim 384). Writes report/mhrag_15articles_overlap512_256.html in
the same table format as the three-arm report. Fresh cache/collection paths so
the original 256/50 run's artifacts are left untouched.
"""

from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from typing import Dict, List

import numpy as np

import vo_config as C
import vo_data as D
import vo_retrieval as VR
import vo_metrics as VM
import vo_three_arms as T  # reuse generation prompts + _index
from embeddings import get_embedder, embedding_signature

VM.KS = C.TOP_K_VALUES

# Configuration-keyed artifacts keep every chunking run reproducible and avoid
# accidentally reusing questions generated for different chunk text.
RUN_TAG = f"{C.CHUNK_SIZE}_{C.CHUNK_OVERLAP}"
LOG_TAG = f"chunk{RUN_TAG}"
GEN_PATH = C.DATA_DIR / f"overlap{C.CHUNK_SIZE}_general_questions.jsonl"
SENT_PATH = C.DATA_DIR / f"overlap{C.CHUNK_SIZE}_sentence3_questions.jsonl"
P8_PATH = C.DATA_DIR / f"overlap{C.CHUNK_SIZE}_promptagator8_questions.jsonl"
D30_PATH = C.DATA_DIR / f"overlap{C.CHUNK_SIZE}_doc2querypp30_questions.jsonl"
P8_EXAMPLES_PATH = C.DATA_DIR / "promptagator_style_examples.json"
BASE_COLL = f"mhrag_vo15_chunk{RUN_TAG}_baseline"
GEN_COLL = f"mhrag_vo15_chunk{RUN_TAG}_general_q"
SENT1_COLL = f"mhrag_vo15_chunk{RUN_TAG}_sentence1_q"
SENT3_COLL = f"mhrag_vo15_chunk{RUN_TAG}_sentence3_q"
P8_COLL = f"mhrag_vo15_chunk{RUN_TAG}_promptagator8_q"
D30_COLL = f"mhrag_vo15_chunk{RUN_TAG}_doc2querypp30_q"
REPORT = (
    C.PROJECT_ROOT / "report" / f"mhrag_15articles_overlap{RUN_TAG}_promptagator.html"
)

# Chunking and gold-alignment artifacts must also be isolated by configuration.
for _attr, _stem, _ext in (
    ("PILOT_ARTICLES", "pilot_15_articles", ".json"),
    ("PILOT_ELIGIBLE", "pilot_eligible_queries", ".jsonl"),
    ("PILOT_EXCLUDED", "pilot_excluded_queries", ".jsonl"),
    ("PILOT_REPORT", "pilot_subset_report", ".json"),
    ("PROCESSED_ARTICLES", "processed_articles", ".jsonl"),
    ("PREPROCESS_REPORT", "preprocessing_report", ".json"),
    ("CHUNKS_PATH", "chunks", ".jsonl"),
    ("GOLD_MAPPING", "gold_chunk_mapping", ".jsonl"),
    ("GOLD_REPORT", "gold_alignment_report", ".json"),
    ("UNRESOLVED_GOLD", "unresolved_gold_evidence", ".jsonl"),
):
    setattr(C, _attr, C.DATA_DIR / f"{_stem}_{RUN_TAG}{_ext}")


# --------------------------------------------------------------------------- #
# Generation (general, sentence-focused, and Promptagator), cached separately
# --------------------------------------------------------------------------- #
_P8_SYSTEM = (
    "You write realistic retrieval questions for a news search system. Follow "
    "the target-query examples only for style; every generated question must be "
    "grounded in the supplied passage. Output ONLY a JSON object."
)


def _style_examples() -> List[dict]:
    """Select 2 examples/type with no query or article overlap with evaluation."""
    if P8_EXAMPLES_PATH.exists():
        return json.load(P8_EXAMPLES_PATH.open(encoding="utf-8"))["examples"]
    eval_ids = {q["query_id"] for q in D.load_eligible_queries()}
    selected_articles = {c["parent_document_id"] for c in D.load_chunks()}
    by_type = {"inference": [], "temporal": [], "comparison": []}
    for q in D.load_all_queries():
        short = q.get("question_type", "").replace("_query", "")
        if (
            short not in by_type
            or q.get("query_id") in eval_ids
            or not q.get("evidence_list")
        ):
            continue
        articles = {e.get(C.ARTICLE_ID_FIELD) for e in q["evidence_list"]}
        if articles & selected_articles:
            continue
        by_type[short].append(q)
    rng = random.Random(C.SEED)
    chosen = []
    for typ in ("inference", "temporal", "comparison"):
        rng.shuffle(by_type[typ])
        chosen.extend(
            {"type": typ, "query_id": q["query_id"], "query": q["query"].strip()}
            for q in by_type[typ][:2]
        )
    if len(chosen) != 6:
        raise RuntimeError("Could not select six leakage-free Promptagator examples")
    json.dump(
        {"seed": C.SEED, "leakage_check": "passed", "examples": chosen},
        P8_EXAMPLES_PATH.open("w", encoding="utf-8"),
        indent=2,
    )
    return chosen


def _gen_promptagator(client, chunk, examples):
    ex_block = "\n".join(f"- [{x['type']}] {x['query']}" for x in examples)
    user = f'''MultiHop-RAG examples:\n{ex_block}\n\nPassage:\n"""\n{chunk["text"]}\n"""\n\nGenerate exactly 8 realistic retrieval questions for this chunk.\n\nMatch the style of the provided MultiHop-RAG examples, including inference,\ntemporal, and comparison questions.\n\nRules:\n- Each question must be answerable from this chunk.\n- Use approximately 8–15 words.\n- Preserve important names, dates, events, and numerical details.\n- Avoid generic, unsupported, or repetitive questions.\n- Do not mention “the chunk” or “the text.”\n\nReturn JSON: {{"questions": ["...", ...]}}'''
    for _ in range(C.GEN_MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=C.gen_model(),
                temperature=C.GEN_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _P8_SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            qs = T._dedup(json.loads(r.choices[0].message.content).get("questions", []))
            if len(qs) == 8:
                return qs
        except Exception:
            continue
    return []


_D30_SYSTEM = (
    "You create grounded retrieval questions for a news search system. First "
    "analyze the supplied article passage into topics and keywords, then write "
    "the requested questions. Output ONLY a JSON object."
)


def _gen_doc2querypp(client, chunk):
    user = f'''Article passage:\n"""\n{chunk["text"]}\n"""\n\nIdentify 2–5 main topics and 8–15 important keywords from this article.\n\nThen generate exactly 30 natural retrieval questions that collectively cover\nthe article’s different topics.\n\nRules:\n- Cover events, entities, dates, comparisons, causes, results, and inferred\n  connections when available.\n- Preserve important names, dates, numbers, and exact terminology.\n- Each question must be answerable from the article.\n- Avoid duplicate questions and repeated coverage of the same fact.\n- Keep the questions concise and realistic.\n\nReturn JSON: {{"topics": ["..."], "keywords": ["..."], "questions": ["...", ...]}}'''
    for _ in range(C.GEN_MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=C.gen_model(),
                temperature=C.GEN_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _D30_SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            data = json.loads(r.choices[0].message.content)
            topics = [str(x).strip() for x in data.get("topics", []) if str(x).strip()]
            keywords = [
                str(x).strip() for x in data.get("keywords", []) if str(x).strip()
            ]
            questions = T._dedup(data.get("questions", []))
            # If near-duplicate removal leaves a small deficit, explicitly ask
            # for only the missing distinct questions rather than accepting a
            # partial record or regenerating the entire set blindly.
            if len(topics) >= 2 and len(keywords) >= 8 and 0 < len(questions) < 30:
                missing = 30 - len(questions)
                existing = "\n".join(f"- {q}" for q in questions)
                repair = client.chat.completions.create(
                    model=C.gen_model(),
                    temperature=C.GEN_TEMPERATURE,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _D30_SYSTEM},
                        {
                            "role": "user",
                            "content": f'''Article passage:\n"""\n{chunk["text"]}\n"""\n\nAlready accepted questions:\n{existing}\n\nGenerate exactly {missing} additional concise retrieval questions. They must be answerable from the passage and must not duplicate or repeat coverage of any accepted question. Return JSON: {{"questions": ["...", ...]}}''',
                        },
                    ],
                )
                questions = T._dedup(
                    questions
                    + json.loads(repair.choices[0].message.content).get("questions", [])
                )
            if len(topics) >= 2 and len(keywords) >= 8 and len(questions) >= 30:
                return {
                    "topics": topics[:5],
                    "keywords": keywords[:15],
                    "questions": questions[:30],
                }
        except Exception:
            continue
    return {"topics": [], "keywords": [], "questions": []}


def generate(force: bool = False) -> None:
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
    pcache = (
        {r["chunk_id"]: r for r in D.read_jsonl(P8_PATH)}
        if (not force and P8_PATH.exists())
        else {}
    )
    dcache = (
        {r["chunk_id"]: r for r in D.read_jsonl(D30_PATH)}
        if (not force and D30_PATH.exists())
        else {}
    )
    todo_g = [c for c in chunks if c["chunk_id"] not in gcache]
    todo_s = [c for c in chunks if c["chunk_id"] not in scache]
    # Empty/partial cached rows are failures, not completed generation.
    todo_p = [
        c
        for c in chunks
        if len(pcache.get(c["chunk_id"], {}).get("questions", [])) != 8
    ]
    todo_d = [
        c
        for c in chunks
        if len(dcache.get(c["chunk_id"], {}).get("questions", [])) != 30
    ]
    examples = _style_examples()
    print(
        f"[{LOG_TAG}] general to-do {len(todo_g)} | sentence to-do {len(todo_s)} | "
        f"P8 to-do {len(todo_p)} | D30 to-do {len(todo_d)}"
    )
    client = C.openai_client()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        fg = {ex.submit(T._gen_general, client, c): c for c in todo_g}
        fs = {ex.submit(T._gen_sentence, client, c): c for c in todo_s}
        fp = {ex.submit(_gen_promptagator, client, c, examples): c for c in todo_p}
        fd = {ex.submit(_gen_doc2querypp, client, c): c for c in todo_d}
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
        for fut in as_completed(list(fp)):
            c = fp[fut]
            pcache[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "questions": fut.result(),
            }
        for fut in as_completed(list(fd)):
            c = fd[fut]
            dcache[c["chunk_id"]] = {"chunk_id": c["chunk_id"], **fut.result()}
    D._write_jsonl(
        GEN_PATH, [gcache[c["chunk_id"]] for c in chunks if c["chunk_id"] in gcache]
    )
    D._write_jsonl(
        SENT_PATH, [scache[c["chunk_id"]] for c in chunks if c["chunk_id"] in scache]
    )
    D._write_jsonl(
        P8_PATH, [pcache[c["chunk_id"]] for c in chunks if c["chunk_id"] in pcache]
    )
    D._write_jsonl(
        D30_PATH, [dcache[c["chunk_id"]] for c in chunks if c["chunk_id"] in dcache]
    )
    ng = sum(len(gcache[c["chunk_id"]]["questions"]) for c in chunks)
    ns = sum(
        len(s["questions"]) for c in chunks for s in scache[c["chunk_id"]]["sentences"]
    )
    np8 = sum(len(pcache[c["chunk_id"]]["questions"]) for c in chunks)
    nd30 = sum(len(dcache[c["chunk_id"]]["questions"]) for c in chunks)
    print(
        f"[{LOG_TAG}] general {ng} Q | sentence {ns} Q | P8 {np8} Q | "
        f"D30 {nd30} Q in {time.perf_counter() - t0:.1f}s"
    )


# --------------------------------------------------------------------------- #
# Indexing (A, B, E1, E3, P8, D30)
# --------------------------------------------------------------------------- #
def build_indexes() -> Dict[str, int]:
    gen = {r["chunk_id"]: r for r in D.read_jsonl(GEN_PATH)}
    sent = {r["chunk_id"]: r for r in D.read_jsonl(SENT_PATH)}
    p8 = {r["chunk_id"]: r for r in D.read_jsonl(P8_PATH)}
    d30 = {r["chunk_id"]: r for r in D.read_jsonl(D30_PATH)}
    chunks = D.load_chunks()

    # Baseline chunk vectors (always rebuilt for this run's embedder).
    embedder = get_embedder()
    coll = C.reset_collection(BASE_COLL)
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
    nb = C.get_collection(BASE_COLL).count()

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

    ng = T._index(GEN_COLL, g_items)
    ns1 = T._index(SENT1_COLL, sent_items(1))
    ns3 = T._index(SENT3_COLL, sent_items(3))
    np8 = T._index(
        P8_COLL,
        [(q, c["chunk_id"]) for c in chunks for q in p8[c["chunk_id"]]["questions"]],
    )
    nd30 = T._index(
        D30_COLL,
        [(q, c["chunk_id"]) for c in chunks for q in d30[c["chunk_id"]]["questions"]],
    )
    print(f"[{LOG_TAG}] vectors: A={nb} B={ng} E1={ns1} E3={ns3} P8={np8} D30={nd30}")
    return {"A": nb, "B": ng, "E1": ns1, "E3": ns3, "P8": np8, "D30": nd30}


# --------------------------------------------------------------------------- #
# Evaluate + report (four arms)
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
    ca = C.get_collection(BASE_COLL)
    cb = C.get_collection(GEN_COLL)
    ce1 = C.get_collection(SENT1_COLL)
    ce3 = C.get_collection(SENT3_COLL)
    cp8 = C.get_collection(P8_COLL)
    cd30 = C.get_collection(D30_COLL)
    R = {k: [] for k in ("A", "B", "E1", "E3", "P8", "D30")}
    colls = {"B": cb, "E1": ce1, "E3": ce3, "P8": cp8, "D30": cd30}
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
        ("E1", "1 question per sentence", PQ["E1"], vecs["E1"]),
        ("E3", "3 questions per sentence", PQ["E3"], vecs["E3"]),
        ("P8", "Promptagator-style 8 questions / chunk", PQ["P8"], vecs["P8"]),
        ("D30", "Doc2Query++ 30 questions / chunk", PQ["D30"], vecs["D30"]),
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
    n_chunks = len(D.load_chunks())
    sig = embedding_signature()
    p8_questions = [q for row in D.read_jsonl(P8_PATH) for q in row["questions"]]
    p8_in_range = sum(
        8 <= len(re.findall(r"\b[\w’'-]+\b", q)) <= 15 for q in p8_questions
    )
    p8_adherence = 100 * p8_in_range / max(len(p8_questions), 1)

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

    caption = (
        f"n={n} evaluation queries · k∈{{1,5,10}} · corpus = {n_chunks} chunks from 15 articles · "
        f"chunking {C.CHUNK_SIZE}-token windows / {C.CHUNK_OVERLAP}-token overlap · embedder {escape(sig)} · generator gpt-5.4-mini. "
        f"P8 generated exactly 8 questions/chunk; {p8_adherence:.1f}% were 8–15 words. "
        f"Each non-baseline cell: value (Δ vs A); green/red/gray = positive/negative/zero; "
        f"<b>*</b> = delta 95% bootstrap CI (1000 resamples) excludes 0. Vectors is the cost column (×A)."
    )
    wtl_line = "Paired per-query, Evidence Recall@5 vs A — " + " &nbsp;·&nbsp; ".join(
        f"<b>{tag}:</b> {wtls[tag][0]} W / {wtls[tag][1]} T / {wtls[tag][2]} L"
        for tag, _n, _P, _v in arms
        if tag != "A"
    )

    doc = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>15-article MultiHop-RAG — {C.CHUNK_SIZE}/{C.CHUNK_OVERLAP} overlap — chunks vs general vs sentence questions</title><style>
:root{{--bg:#fff;--fg:#161a21;--muted:#5b6572;--line:#e3e7ee;--card:#f7f9fc;--good:#0a7d3f;--bad:#c0392b;--flat:#8792a1;--hl:#fff4d1;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#242b36;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#8792a1;--hl:#39310c;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:34px 22px 70px}}h1{{font-size:23px;margin:0 0 4px}}h2{{font-size:18px;margin:30px 0 8px}}
.cap{{color:var(--muted);font-size:12.5px;margin:6px 0 12px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}th,td{{border:1px solid var(--line);padding:8px;text-align:center;vertical-align:middle}}
th{{background:var(--card);font-weight:600}}td.nm,th:first-child{{text-align:left}}tr.base{{background:var(--card)}}tr.base td.nm{{font-weight:700}}
.v{{font-weight:700;font-size:14px}}.ci{{font-size:10px;color:var(--muted)}}.d{{font-size:11.5px;margin-top:1px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}
.wtl{{font-size:12.5px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:6px 0}}
.leg{{font-size:12px;color:var(--muted);margin-top:8px}}
</style></head><body><div class="wrap">
<h1>15-article MultiHop-RAG — {C.CHUNK_SIZE}/{C.CHUNK_OVERLAP} chunking — chunks vs general vs sentence questions</h1>
<p class="cap">Re-chunked at {C.CHUNK_SIZE}-token windows with {C.CHUNK_OVERLAP}-token overlap. Does indexing generated questions instead of the original chunks retrieve gold evidence better? Dense cosine, local ChromaDB, gpt-5.4-mini. Rows = baseline A first, then each arm.</p>
{tbl}
<p class="cap">{caption}</p>
<div class="wtl">{wtl_line}</div>
<p class="leg">Only asterisked deltas are statistically distinguishable from zero; unmarked deltas are not described as improvements or regressions. Closed 15-article pilot — not full-corpus MultiHop-RAG.</p>
</div></body></html>"""
    REPORT.write_text(doc, encoding="utf-8")
    print(f"[{LOG_TAG}] wrote {REPORT}")


def run(force_gen: bool = True):
    D.build_all(force=False)
    generate(force=force_gen)
    vecs = build_indexes()
    METS, arms, cd, wtls, n = evaluate(vecs)
    render(METS, arms, cd, wtls, n)
    for mk, lbl in METS:
        vv = "  ".join(f"{tag}={cd[mk][tag]['mean']:.3f}" for tag, _n, _P, _v in arms)
        print(f"  {lbl:<22} {vv}")


if __name__ == "__main__":
    run()
