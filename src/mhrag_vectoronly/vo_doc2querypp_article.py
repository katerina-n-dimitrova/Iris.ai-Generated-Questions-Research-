"""
Corrected article-level Doc2Query++ experiment on the closed 15-article pilot.

Generation unit: complete cleaned article (30 Q/A/evidence records per article).
Each verbatim evidence span is mapped to a containing 512/256 chunk. Retrieval
keeps chunk and generated-question indexes separate, min-max normalizes their
per-query chunk scores, and evaluates weighted score fusion.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path
from typing import Dict, List

import numpy as np

import vo_overlap as V
import vo_metrics as VM
from embeddings import embedding_signature, get_embedder

if (V.C.CHUNK_SIZE, V.C.CHUNK_OVERLAP) != (512, 256):
    raise RuntimeError("Article Doc2Query++ requires chunking 512/256")

VM.KS = V.C.TOP_K_VALUES

DATA_PATH = V.C.DATA_DIR / "doc2querypp_article30_grounded.jsonl"
RESULTS_PATH = V.C.RESULTS_DIR / "doc2querypp_article30_fusion.json"
QUESTION_COLL = "mhrag_vo15_chunk512_256_doc2querypp_article30_q"
REPORT = V.REPORT
WEIGHTS = (0.3, 0.5, 0.7, 1.0)
MARKER_START = "<!-- DOC2QUERYPP_ARTICLE_START -->"
MARKER_END = "<!-- DOC2QUERYPP_ARTICLE_END -->"

_SYSTEM = (
    "You create grounded retrieval training examples from a complete news "
    "article. Every evidence string must be copied verbatim from the article. "
    "Output ONLY a JSON object."
)


def _articles() -> List[dict]:
    return V.D.read_jsonl(V.C.PROCESSED_ARTICLES)


def _chunks_by_article() -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for c in V.D.load_chunks():
        out.setdefault(c["parent_document_id"], []).append(c)
    return out


def _norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _assign_chunk(evidence: str, chunks: List[dict]) -> str | None:
    """Choose a containing chunk, preferring evidence furthest from boundaries."""
    candidates = []
    for c in chunks:
        pos = c["text"].find(evidence)
        if pos >= 0:
            margin = min(pos, len(c["text"]) - pos - len(evidence))
            candidates.append((margin, -c["chunk_position"], c["chunk_id"]))
    return max(candidates)[2] if candidates else None


def _parse_items(data: dict, article: dict, chunks: List[dict]) -> List[dict]:
    body = article["cleaned_body"]
    out, seen = [], set()
    for raw in data.get("items", []):
        q = str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        evidence = str(raw.get("evidence", "")).strip()
        key = _norm_question(q)
        if not q or not answer or not evidence or key in seen:
            continue
        # "Exact supporting evidence" is enforced, not fuzzy-matched.
        if evidence not in body:
            continue
        cid = _assign_chunk(evidence, chunks)
        if not cid:
            continue
        seen.add(key)
        out.append(
            {
                "question": q,
                "answer": answer,
                "evidence": evidence,
                "parent_article_id": article["article_id"],
                "parent_chunk_id": cid,
            }
        )
    return out


def _initial_prompt(article: dict) -> str:
    return f'''Complete article:
"""
{article["cleaned_body"]}
"""

Identify 2–5 main topics and 8–15 important keywords from this complete article.

Then generate exactly 30 concise, natural retrieval questions for the article.
The questions must collectively cover different topics, entities, events, dates,
comparisons, causes, consequences, and important keywords when available.

Rules:
- Avoid repetitive questions focused only on the article's main topic.
- Every question must be answerable from the article.
- Save a concise answer for every question.
- Save one short exact supporting-evidence span copied VERBATIM from the article.
- Evidence should be sufficiently short to fit inside one retrieval chunk.

Return JSON:
{{"topics":["2–5 topics"],"keywords":["8–15 keywords"],
  "items":[{{"question":"...","answer":"...","evidence":"verbatim quote"}}, ...]}}'''


def _call(client, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=V.C.gen_model(),
        temperature=V.C.GEN_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


def _generate_article(client, article: dict, chunks: List[dict]) -> dict:
    topics: List[str] = []
    keywords: List[str] = []
    items: List[dict] = []

    # Full regeneration attempts always use the complete article as the unit.
    for _ in range(V.C.GEN_MAX_RETRIES):
        try:
            data = _call(client, _initial_prompt(article))
            topics = [str(x).strip() for x in data.get("topics", []) if str(x).strip()][
                :5
            ]
            keywords = [
                str(x).strip() for x in data.get("keywords", []) if str(x).strip()
            ][:15]
            items = _parse_items(data, article, chunks)
            if len(topics) >= 2 and len(keywords) >= 8 and len(items) >= 30:
                break
        except Exception:
            continue

    # Repair only missing grounded records, still providing the complete article.
    for _ in range(4):
        if len(topics) >= 2 and len(keywords) >= 8 and len(items) >= 30:
            break
        missing = max(1, 30 - len(items))
        existing = "\n".join(f"- {x['question']}" for x in items)
        prompt = f'''Complete article:
"""
{article["cleaned_body"]}
"""

Existing accepted questions:
{existing or "(none)"}

Generate {missing + 3} additional distinct retrieval question records so that at
least {missing} survive validation. Do not repeat an existing question or fact.
Each record needs a concise answer and a short evidence span copied VERBATIM from
the complete article. Return JSON:
{{"topics":["2–5 topics"],"keywords":["8–15 keywords"],
  "items":[{{"question":"...","answer":"...","evidence":"verbatim quote"}}, ...]}}'''
        try:
            data = _call(client, prompt)
            if len(topics) < 2:
                topics = [
                    str(x).strip() for x in data.get("topics", []) if str(x).strip()
                ][:5]
            if len(keywords) < 8:
                keywords = [
                    str(x).strip() for x in data.get("keywords", []) if str(x).strip()
                ][:15]
            merged = {"items": items + data.get("items", [])}
            items = _parse_items(merged, article, chunks)
        except Exception:
            continue

    return {
        "article_id": article["article_id"],
        "title": article["title"],
        "topics": topics[:5],
        "keywords": keywords[:15],
        "items": items[:30],
        "valid": len(topics) >= 2 and len(keywords) >= 8 and len(items) >= 30,
    }


def generate(force: bool = False) -> List[dict]:
    articles = _articles()
    chunks = _chunks_by_article()
    cache = (
        {r["article_id"]: r for r in V.D.read_jsonl(DATA_PATH)}
        if DATA_PATH.exists() and not force
        else {}
    )
    todo = [
        a
        for a in articles
        if not cache.get(a["article_id"], {}).get("valid")
        or len(cache[a["article_id"]].get("items", [])) != 30
    ]
    print(f"[article-d30] articles to-do {len(todo)} / {len(articles)}")
    if todo:
        client = V.C.openai_client()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                ex.submit(_generate_article, client, a, chunks[a["article_id"]]): a
                for a in todo
            }
            for fut in as_completed(futures):
                a = futures[fut]
                cache[a["article_id"]] = fut.result()
        print(f"[article-d30] generation {time.perf_counter() - t0:.1f}s")
    rows = [cache[a["article_id"]] for a in articles]
    V.D._write_jsonl(DATA_PATH, rows)
    valid = sum(r["valid"] and len(r["items"]) == 30 for r in rows)
    print(
        f"[article-d30] valid articles {valid}/{len(rows)} | "
        f"questions {sum(len(r['items']) for r in rows)}"
    )
    return rows


def build_question_index(rows: List[dict]) -> int:
    items = [item for row in rows for item in row["items"]]
    if len(items) != 30 * len(rows):
        raise RuntimeError("Cannot index: article-level D30 cache is incomplete")
    embedder = get_embedder()
    vectors = embedder.embed_documents([x["question"] for x in items])
    coll = V.C.reset_collection(QUESTION_COLL)
    ids, docs, metas = [], [], []
    for i, item in enumerate(items):
        qid = f"article-d30::{i}"
        ids.append(qid)
        docs.append(item["question"])
        metas.append(
            {
                "generated_question_id": qid,
                "parent_article_id": item["parent_article_id"],
                "parent_document_id": item["parent_article_id"],
                "parent_chunk_id": item["parent_chunk_id"],
                "answer": item["answer"],
                "evidence": item["evidence"],
            }
        )
    coll.add(ids=ids, embeddings=vectors, documents=docs, metadatas=metas)
    print(f"[article-d30] indexed {len(ids)} grounded questions")
    return len(ids)


def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _all_chunk_scores(coll, qvec) -> Dict[str, float]:
    n = coll.count()
    res = coll.query(
        query_embeddings=[qvec], n_results=n, include=["metadatas", "distances"]
    )
    return {
        m["parent_chunk_id"]: 1.0 - float(d)
        for m, d in zip(res["metadatas"][0], res["distances"][0])
    }


def _all_question_scores(coll, qvec) -> Dict[str, float]:
    n = coll.count()
    res = coll.query(
        query_embeddings=[qvec], n_results=n, include=["metadatas", "distances"]
    )
    best: Dict[str, float] = {}
    for m, d in zip(res["metadatas"][0], res["distances"][0]):
        cid, score = m["parent_chunk_id"], 1.0 - float(d)
        if cid not in best or score > best[cid]:
            best[cid] = score
    return best


def _rank(scores: Dict[str, float], chunk_map: Dict[str, dict]) -> List[dict]:
    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[: V.C.RANK_DEPTH]
    return [
        {
            "rank": i,
            "chunk_id": cid,
            "parent_document_id": chunk_map[cid]["parent_document_id"],
            "score": round(float(score), 6),
        }
        for i, (cid, score) in enumerate(ordered, 1)
    ]


def _common(q: dict, gold: dict) -> dict:
    return {
        "query_id": q["query_id"],
        "question_type": q["question_type"],
        "n_required_documents": q["n_required_documents"],
        "n_required_evidence_facts": q["n_required_evidence_facts"],
        "gold_chunk_ids": gold["gold_chunk_ids"],
        "evidence_units": gold["evidence_units"],
        "required_article_ids": q["required_article_ids"],
    }


def retrieve() -> Dict[str, List[dict]]:
    queries, gold = V.D.load_eligible_queries(), V.D.load_gold()
    chunk_map = {c["chunk_id"]: c for c in V.D.load_chunks()}
    embedder = get_embedder()
    chunk_coll = V.C.get_collection(V.BASE_COLL)
    old_d30_coll = V.C.get_collection(V.D30_COLL)
    article_q_coll = V.C.get_collection(QUESTION_COLL)
    conditions = {"A": [], "D30-old": [], "F30": [], "F50": [], "F70": [], "Q100": []}

    for q in queries:
        qvec = embedder.embed_query(q["query"])
        cs_raw = _all_chunk_scores(chunk_coll, qvec)
        qs_raw = _all_question_scores(article_q_coll, qvec)
        cs, qs = _minmax(cs_raw), _minmax(qs_raw)
        common = _common(q, gold[q["query_id"]])
        conditions["A"].append({**common, "ranked": _rank(cs_raw, chunk_map)})

        old = V.VR.retrieve_generated(
            old_d30_coll, qvec, V.C.RANK_DEPTH, V.C.CANDIDATE_MULTIPLIER
        )
        conditions["D30-old"].append({**common, "ranked": old})

        for weight, tag in ((0.3, "F30"), (0.5, "F50"), (0.7, "F70"), (1.0, "Q100")):
            fused = {
                cid: weight * qs.get(cid, 0.0) + (1.0 - weight) * cs.get(cid, 0.0)
                for cid in chunk_map
            }
            conditions[tag].append({**common, "ranked": _rank(fused, chunk_map)})
    return conditions


def evaluate(rankings: Dict[str, List[dict]], n_vectors: int) -> dict:
    per = {
        tag: {r["query_id"]: VM.per_query(r) for r in rows}
        for tag, rows in rankings.items()
    }
    qids = sorted(per["A"])
    metrics = [
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ]

    def vals(tag, key):
        return np.array([per[tag][qid][key] for qid in qids])

    data = {}
    for key, _ in metrics:
        base = vals("A", key)
        data[key] = {}
        for tag in per:
            x = vals(tag, key)
            lo, hi = V._ci(x)
            item = {"mean": float(x.mean()), "ci_low": lo, "ci_high": hi}
            if tag != "A":
                dm, dlo, dhi, significant = V._dci(base, x)
                item.update(
                    delta=float(dm),
                    delta_low=dlo,
                    delta_high=dhi,
                    significant=bool(significant),
                )
            data[key][tag] = item

    # Primary selection: highest ER@5, then MRR@10, then lower question weight.
    fusion_tags = ["F30", "F50", "F70", "Q100"]
    weight_order = {"F30": 0.3, "F50": 0.5, "F70": 0.7, "Q100": 1.0}
    best = max(
        fusion_tags,
        key=lambda t: (
            data["evidence_recall@5"][t]["mean"],
            data["mrr@10"][t]["mean"],
            -weight_order[t],
        ),
    )
    result = {
        "n_queries": len(qids),
        "n_articles": 15,
        "n_chunks": len(V.D.load_chunks()),
        "n_article_questions": n_vectors,
        "normalization": "per-query min-max, independently per index",
        "parent_question_aggregation": "max cosine",
        "weights": {"F30": 0.3, "F50": 0.5, "F70": 0.7, "Q100": 1.0},
        "best_fusion": best,
        "metrics": data,
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _cell(item: dict, base: bool = False) -> str:
    if base:
        return (
            f'<div class="v">{item["mean"]:.3f}</div>'
            f'<div class="ci">[{item["ci_low"]:.3f}, {item["ci_high"]:.3f}]</div>'
        )
    delta = item["delta"]
    cls = "good" if delta > 1e-9 else ("bad" if delta < -1e-9 else "flat")
    star = "<b>*</b>" if item["significant"] else ""
    return (
        f'<div class="v">{item["mean"]:.3f}</div>'
        f'<div class="d {cls}" title="Δ 95% CI '
        f'[{item["delta_low"]:+.3f}, {item["delta_high"]:+.3f}]">'
        f"({delta:+.3f}{star})</div>"
    )


def update_report(result: dict) -> None:
    metrics = [
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ]
    rows = [
        ("A", "Original chunks only", 119),
        ("D30-old", "Previous chunk-level D30 question-only", 3570),
        ("F30", "Corrected dual index · 0.3 question / 0.7 chunk", "119 + 450"),
        ("F50", "Corrected dual index · 0.5 question / 0.5 chunk", "119 + 450"),
        ("F70", "Corrected dual index · 0.7 question / 0.3 chunk", "119 + 450"),
        ("Q100", "Corrected article D30 · question-only", 450),
    ]
    body = ""
    for tag, label, vecs in rows:
        cells = "".join(
            f"<td>{_cell(result['metrics'][key][tag], tag == 'A')}</td>"
            for key, _ in metrics
        )
        best = " <b>· best fusion</b>" if tag == result["best_fusion"] else ""
        body += (
            f'<tr class="{"base" if tag == "A" else ""}">'
            f'<td class="nm"><b>{escape(tag)}</b> · {escape(label)}{best}</td>'
            f"{cells}<td>{vecs}</td></tr>"
        )
    heads = "".join(f"<th>{escape(label)}</th>" for _, label in metrics)
    section = f"""{MARKER_START}
<h2>Corrected article-level Doc2Query++ — separate indexes and score fusion</h2>
<p class="cap">Generation unit = complete article. Each of 450 questions stores an
answer, verbatim supporting evidence, parent article ID, and evidence-containing
parent chunk ID. Chunk and question embeddings remain in separate indexes.
Scores are min–max normalized independently per query, then fused. D30-old is
the earlier, incorrectly chunk-generated question-only control.</p>
<table><thead><tr><th>Condition</th>{heads}<th>Index vectors</th></tr></thead>
<tbody>{body}</tbody></table>
<p class="cap">Best fusion by Evidence Recall@5 (MRR@10 tie-break):
<b>{escape(result["best_fusion"])}</b>. n={result["n_queries"]} queries ·
{result["n_chunks"]} chunks · 15 articles · 512/256 chunking ·
embedder {escape(embedding_signature())}. Each delta is versus A; * means its
paired 95% bootstrap interval excludes zero.</p>
{MARKER_END}"""

    html = REPORT.read_text(encoding="utf-8")
    if MARKER_START in html:
        start = html.index(MARKER_START)
        end = html.index(MARKER_END, start) + len(MARKER_END)
        html = html[:start] + section + html[end:]
    else:
        html = html.replace("</div></body></html>", section + "\n</div></body></html>")
    REPORT.write_text(html, encoding="utf-8")
    print(f"[article-d30] updated {REPORT}")


def run(force_gen: bool = False) -> dict:
    V.D.build_all(force=False)
    rows = generate(force_gen)
    if not all(r["valid"] and len(r["items"]) == 30 for r in rows):
        raise RuntimeError("Article-level generation incomplete; rerun to repair cache")
    n_vectors = build_question_index(rows)
    rankings = retrieve()
    result = evaluate(rankings, n_vectors)
    update_report(result)
    for key, label in [
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ]:
        print(
            f"  {label:<22} "
            + "  ".join(
                f"{tag}={result['metrics'][key][tag]['mean']:.3f}"
                for tag in ("A", "D30-old", "F30", "F50", "F70", "Q100")
            )
        )
    print(f"[article-d30] best fusion {result['best_fusion']}")
    return result


if __name__ == "__main__":
    run()
