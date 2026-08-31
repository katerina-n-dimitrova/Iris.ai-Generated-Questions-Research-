"""
Doc2Query--- filtering experiment over existing chunk-generated questions.

No retrieval questions are generated. Existing B, E3, and old chunk-level D30
questions/embeddings are filtered using:
  1) LLM-judged answerability from parent chunk,
  2) LLM-judged chunk specificity,
  3) LLM-judged absence of unsupported information,
  4) deterministic within-parent near-duplicate removal,
  5) parent chunk in top 5 when the question searches the original chunk index.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Dict, List

import numpy as np

import vo_overlap as V
import vo_metrics as VM
from embeddings import embedding_signature, get_embedder

if (V.C.CHUNK_SIZE, V.C.CHUNK_OVERLAP) != (512, 256):
    raise RuntimeError("Filtering experiment requires chunking 512/256")

VM.KS = V.C.TOP_K_VALUES

AUDIT_PATH = V.C.DATA_DIR / "doc2query_filter_audit.jsonl"
RESULTS_PATH = V.C.RESULTS_DIR / "doc2query_filter_results.json"
REPORT = V.REPORT
MARKER_START = "<!-- DOC2QUERY_FILTER_START -->"
MARKER_END = "<!-- DOC2QUERY_FILTER_END -->"
JUDGE_BATCH = 60
DEDUP_THRESHOLD = 0.92

SOURCES = {
    "B": V.GEN_COLL,
    "E3": V.SENT3_COLL,
    "D30": V.D30_COLL,
}
FILTERED_COLLS = {
    "B": "mhrag_vo15_chunk512_256_b_filtered",
    "E3": "mhrag_vo15_chunk512_256_e3_filtered",
    "D30": "mhrag_vo15_chunk512_256_d30_filtered",
}

_JUDGE_SYSTEM = (
    "You are a strict retrieval-question quality auditor. You do not write or "
    "rewrite questions. Judge each supplied question only against its parent "
    "chunk and return JSON."
)


def _load_existing() -> Dict[str, List[dict]]:
    """Load existing text, metadata, and embeddings without re-embedding."""
    out = {}
    for tag, name in SOURCES.items():
        coll = V.C.get_collection(name)
        raw = coll.get(include=["documents", "metadatas", "embeddings"])
        rows = []
        for i, (qid, text, meta, emb) in enumerate(
            zip(raw["ids"], raw["documents"], raw["metadatas"], raw["embeddings"])
        ):
            rows.append(
                {
                    "condition": tag,
                    "question_id": qid,
                    "question": text,
                    "parent_chunk_id": meta["parent_chunk_id"],
                    "parent_document_id": meta["parent_document_id"],
                    "embedding": list(emb),
                    "source_order": i,
                }
            )
        out[tag] = rows
    return out


def _judge_prompt(chunk: str, rows: List[dict]) -> str:
    questions = "\n".join(f"{i}. {r['question']}" for i, r in enumerate(rows))
    return f'''Parent chunk:
"""
{chunk}
"""

Questions to audit:
{questions}

For every numbered question, judge these three independent requirements:
- answerable: fully answerable using only the parent chunk;
- specific: specific enough that its wording identifies this chunk rather than
  broadly matching many unrelated news chunks;
- supported: contains no claim, premise, entity relationship, date, or detail
  unsupported by the parent chunk.

Do not create, rewrite, or repair any question. Return JSON only:
{{"judgments":[{{"id":0,"answerable":true,"specific":true,"supported":true}}, ...]}}'''


def _judge_batch(client, chunk: str, rows: List[dict]) -> List[dict]:
    for _ in range(V.C.GEN_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=V.C.gen_model(),
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": _judge_prompt(chunk, rows)},
                ],
            )
            data = json.loads(response.choices[0].message.content)
            by_id = {int(x["id"]): x for x in data.get("judgments", [])}
            if all(i in by_id for i in range(len(rows))):
                return [
                    {
                        "answerable": bool(by_id[i].get("answerable")),
                        "specific": bool(by_id[i].get("specific")),
                        "supported": bool(by_id[i].get("supported")),
                    }
                    for i in range(len(rows))
                ]
        except Exception:
            continue
    return [
        {
            "answerable": False,
            "specific": False,
            "supported": False,
            "judge_failed": True,
        }
        for _ in rows
    ]


def judge(rows_by_tag: Dict[str, List[dict]]) -> Dict[str, dict]:
    chunks = {c["chunk_id"]: c for c in V.D.load_chunks()}
    cached = {}
    if AUDIT_PATH.exists():
        cached = {r["audit_id"]: r for r in V.D.read_jsonl(AUDIT_PATH)}

    tasks = []
    for tag, rows in rows_by_tag.items():
        grouped = defaultdict(list)
        for row in rows:
            aid = f"{tag}::{row['question_id']}"
            if aid not in cached or cached[aid].get("judge_failed"):
                grouped[row["parent_chunk_id"]].append(row)
        for cid, group in grouped.items():
            for i in range(0, len(group), JUDGE_BATCH):
                tasks.append((tag, cid, group[i : i + JUDGE_BATCH]))

    print(f"[filter] judge batches to-do {len(tasks)}")
    if tasks:
        client = V.C.openai_client()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {
                ex.submit(_judge_batch, client, chunks[cid]["text"], batch): (
                    tag,
                    cid,
                    batch,
                )
                for tag, cid, batch in tasks
            }
            for fut in as_completed(futures):
                tag, _cid, batch = futures[fut]
                judgments = fut.result()
                for row, verdict in zip(batch, judgments):
                    aid = f"{tag}::{row['question_id']}"
                    cached[aid] = {
                        "audit_id": aid,
                        "condition": tag,
                        "question_id": row["question_id"],
                        "question": row["question"],
                        "parent_chunk_id": row["parent_chunk_id"],
                        "parent_document_id": row["parent_document_id"],
                        **verdict,
                    }
        print(f"[filter] judging {time.perf_counter() - t0:.1f}s")
    return cached


def self_retrieval(rows_by_tag: Dict[str, List[dict]], audit: Dict[str, dict]) -> None:
    """Criterion 5 using the already-stored question embeddings."""
    base = V.C.get_collection(V.BASE_COLL)
    for tag, rows in rows_by_tag.items():
        for i in range(0, len(rows), 128):
            batch = rows[i : i + 128]
            result = base.query(
                query_embeddings=[r["embedding"] for r in batch],
                n_results=5,
                include=["metadatas", "distances"],
            )
            for row, metas, dists in zip(
                batch, result["metadatas"], result["distances"]
            ):
                parents = [m["parent_chunk_id"] for m in metas]
                aid = f"{tag}::{row['question_id']}"
                rank = (
                    parents.index(row["parent_chunk_id"]) + 1
                    if row["parent_chunk_id"] in parents
                    else None
                )
                audit[aid]["parent_top5"] = rank is not None
                audit[aid]["parent_rank"] = rank
                audit[aid]["parent_score"] = (
                    1.0 - float(dists[rank - 1]) if rank else None
                )


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def deduplicate_and_finalize(
    rows_by_tag: Dict[str, List[dict]], audit: Dict[str, dict]
) -> None:
    """Keep the strongest self-retrieving representative per near-dup cluster."""
    for tag, rows in rows_by_tag.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["parent_chunk_id"]].append(row)
        for _cid, group in grouped.items():

            def passes_other(row):
                a = audit[f"{tag}::{row['question_id']}"]
                return (
                    a["answerable"]
                    and a["specific"]
                    and a["supported"]
                    and a.get("parent_top5", False)
                )

            # Prefer a representative that passes every other criterion, then
            # better self-retrieval and original deterministic order.
            ordered = sorted(
                group,
                key=lambda r: (
                    not passes_other(r),
                    audit[f"{tag}::{r['question_id']}"].get("parent_rank") or 999,
                    -(
                        audit[f"{tag}::{r['question_id']}"].get("parent_score")
                        or -999.0
                    ),
                    r["source_order"],
                ),
            )
            representatives: List[dict] = []
            for row in ordered:
                text = _norm(row["question"])
                duplicate = any(
                    SequenceMatcher(None, text, _norm(x["question"])).ratio()
                    >= DEDUP_THRESHOLD
                    for x in representatives
                )
                a = audit[f"{tag}::{row['question_id']}"]
                a["not_near_duplicate"] = not duplicate
                if not duplicate:
                    representatives.append(row)
            for row in group:
                a = audit[f"{tag}::{row['question_id']}"]
                a.setdefault("not_near_duplicate", False)
                a["kept"] = bool(
                    a["answerable"]
                    and a["specific"]
                    and a["supported"]
                    and a.get("parent_top5", False)
                    and a["not_near_duplicate"]
                )


def save_audit(audit: Dict[str, dict]) -> None:
    V.D._write_jsonl(AUDIT_PATH, [audit[k] for k in sorted(audit)])


def build_filtered_indexes(
    rows_by_tag: Dict[str, List[dict]], audit: Dict[str, dict]
) -> Dict[str, int]:
    counts = {}
    for tag, rows in rows_by_tag.items():
        kept = [r for r in rows if audit[f"{tag}::{r['question_id']}"]["kept"]]
        coll = V.C.reset_collection(FILTERED_COLLS[tag])
        if kept:
            coll.add(
                ids=[f"{tag}-filtered::{i}" for i in range(len(kept))],
                embeddings=[r["embedding"] for r in kept],
                documents=[r["question"] for r in kept],
                metadatas=[
                    {
                        "generated_question_id": r["question_id"],
                        "parent_chunk_id": r["parent_chunk_id"],
                        "parent_document_id": r["parent_document_id"],
                    }
                    for r in kept
                ],
            )
        counts[tag] = len(kept)
    print("[filter] retained", counts)
    return counts


def retrieve(rows_by_tag: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    queries, gold = V.D.load_eligible_queries(), V.D.load_gold()
    embedder = get_embedder()
    base = V.C.get_collection(V.BASE_COLL)
    source_colls = {tag: V.C.get_collection(name) for tag, name in SOURCES.items()}
    filtered_colls = {
        tag: V.C.get_collection(name) for tag, name in FILTERED_COLLS.items()
    }
    conditions = {"A": []}
    for tag in SOURCES:
        conditions[tag] = []
        conditions[f"{tag}-filtered"] = []

    for q in queries:
        qvec = embedder.embed_query(q["query"])
        g = gold[q["query_id"]]
        common = {
            "query_id": q["query_id"],
            "question_type": q["question_type"],
            "n_required_documents": q["n_required_documents"],
            "n_required_evidence_facts": q["n_required_evidence_facts"],
            "gold_chunk_ids": g["gold_chunk_ids"],
            "evidence_units": g["evidence_units"],
            "required_article_ids": q["required_article_ids"],
        }
        conditions["A"].append(
            {
                **common,
                "ranked": V.VR.retrieve_baseline(base, qvec, V.C.RANK_DEPTH),
            }
        )
        for tag in SOURCES:
            conditions[tag].append(
                {
                    **common,
                    "ranked": V.VR.retrieve_generated(
                        source_colls[tag],
                        qvec,
                        V.C.RANK_DEPTH,
                        V.C.CANDIDATE_MULTIPLIER,
                    ),
                }
            )
            conditions[f"{tag}-filtered"].append(
                {
                    **common,
                    "ranked": V.VR.retrieve_generated(
                        filtered_colls[tag],
                        qvec,
                        V.C.RANK_DEPTH,
                        V.C.CANDIDATE_MULTIPLIER,
                    ),
                }
            )
    return conditions


def evaluate(
    rankings: Dict[str, List[dict]],
    counts: Dict[str, int],
    totals: Dict[str, int],
    audit: Dict[str, dict],
) -> dict:
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
        data[key] = {}
        for tag in per:
            x = vals(tag, key)
            lo, hi = V._ci(x)
            item = {"mean": float(x.mean()), "ci_low": lo, "ci_high": hi}
            if tag != "A":
                # Filtered conditions are compared with their own unfiltered arm.
                ref_tag = (
                    tag.removesuffix("-filtered") if tag.endswith("-filtered") else "A"
                )
                ref = vals(ref_tag, key)
                dm, dlo, dhi, sig = V._dci(ref, x)
                item.update(
                    delta=float(dm),
                    delta_low=dlo,
                    delta_high=dhi,
                    significant=bool(sig),
                    reference=ref_tag,
                )
            data[key][tag] = item

    reasons = {}
    for tag in SOURCES:
        rows = [a for a in audit.values() if a["condition"] == tag]
        reasons[tag] = {
            "not_answerable": sum(not a["answerable"] for a in rows),
            "not_specific": sum(not a["specific"] for a in rows),
            "unsupported": sum(not a["supported"] for a in rows),
            "near_duplicate": sum(not a["not_near_duplicate"] for a in rows),
            "parent_not_top5": sum(not a["parent_top5"] for a in rows),
        }
    result = {
        "n_queries": len(qids),
        "totals": totals,
        "retained": counts,
        "removed": {t: totals[t] - counts[t] for t in totals},
        "retained_pct": {t: 100 * counts[t] / totals[t] for t in totals},
        "removal_reasons_nonexclusive": reasons,
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
    d = item["delta"]
    cls = "good" if d > 1e-9 else ("bad" if d < -1e-9 else "flat")
    star = "<b>*</b>" if item["significant"] else ""
    return (
        f'<div class="v">{item["mean"]:.3f}</div>'
        f'<div class="d {cls}" title="Δ vs {item["reference"]} 95% CI '
        f'[{item["delta_low"]:+.3f}, {item["delta_high"]:+.3f}]">'
        f"({d:+.3f}{star})</div>"
    )


def update_report(result: dict) -> None:
    metrics = [
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ]
    rows = [("A", "Original chunks", 119, None)]
    for tag, label in (
        ("B", "General 10 questions / chunk"),
        ("E3", "3 questions / sentence"),
        ("D30", "Previous chunk-level Doc2Query++"),
    ):
        rows.append((tag, label, result["totals"][tag], None))
        rows.append(
            (
                f"{tag}-filtered",
                f"{label} · filtered",
                result["retained"][tag],
                result["retained_pct"][tag],
            )
        )

    body = ""
    for tag, label, vectors, pct in rows:
        cells = "".join(
            f"<td>{_cell(result['metrics'][key][tag], tag == 'A')}</td>"
            for key, _ in metrics
        )
        vtext = (
            str(vectors)
            if pct is None
            else f"{vectors}<div class='d flat'>({pct:.1f}% kept)</div>"
        )
        body += (
            f'<tr class="{"base" if tag == "A" else ""}">'
            f'<td class="nm"><b>{escape(tag)}</b> · {escape(label)}</td>'
            f"{cells}<td>{vtext}</td></tr>"
        )
    heads = "".join(f"<th>{escape(label)}</th>" for _, label in metrics)

    summary_rows = "".join(
        f"<tr><td class='nm'><b>{tag}</b></td>"
        f"<td>{result['totals'][tag]}</td><td>{result['retained'][tag]}</td>"
        f"<td>{result['removed'][tag]}</td>"
        f"<td>{result['retained_pct'][tag]:.1f}%</td>"
        f"<td>{result['removal_reasons_nonexclusive'][tag]['not_answerable']}</td>"
        f"<td>{result['removal_reasons_nonexclusive'][tag]['not_specific']}</td>"
        f"<td>{result['removal_reasons_nonexclusive'][tag]['unsupported']}</td>"
        f"<td>{result['removal_reasons_nonexclusive'][tag]['near_duplicate']}</td>"
        f"<td>{result['removal_reasons_nonexclusive'][tag]['parent_not_top5']}</td></tr>"
        for tag in ("B", "E3", "D30")
    )
    section = f"""{MARKER_START}
<h2>Doc2Query--- filtering — existing chunk-generated questions</h2>
<p class="cap">No questions were generated or rewritten. Existing B, E3, and
chunk-level D30 questions were retained only when fully answerable, chunk-specific,
supported, non-duplicate, and self-retrieving (parent chunk in original chunk
index top 5). Filtered deltas below are versus each arm's own unfiltered version.</p>
<table><thead><tr><th>Condition</th>{heads}<th>Vectors</th></tr></thead>
<tbody>{body}</tbody></table>
<h2>Filtering retention</h2>
<table><thead><tr><th>Source</th><th>Total</th><th>Kept</th><th>Removed</th>
<th>Retained</th><th>Not answerable</th><th>Not specific</th><th>Unsupported</th>
<th>Near duplicate</th><th>Parent not top 5</th></tr></thead>
<tbody>{summary_rows}</tbody></table>
<p class="cap">Removal-reason columns are non-exclusive. Quality criteria 1–3
were judged against the parent chunk at temperature 0; near-duplicates used a
0.92 similarity threshold; self-retrieval used the existing stored question
embeddings. n={result["n_queries"]} queries · 512/256 chunking ·
{escape(embedding_signature())}. * means the paired 95% bootstrap interval
against the corresponding unfiltered arm excludes zero.</p>
{MARKER_END}"""
    html = REPORT.read_text(encoding="utf-8")
    if MARKER_START in html:
        start = html.index(MARKER_START)
        end = html.index(MARKER_END, start) + len(MARKER_END)
        html = html[:start] + section + html[end:]
    else:
        html = html.replace("</div></body></html>", section + "\n</div></body></html>")
    REPORT.write_text(html, encoding="utf-8")
    print(f"[filter] updated {REPORT}")


def run() -> dict:
    V.D.build_all(force=False)
    rows_by_tag = _load_existing()
    totals = {tag: len(rows) for tag, rows in rows_by_tag.items()}
    audit = judge(rows_by_tag)
    self_retrieval(rows_by_tag, audit)
    deduplicate_and_finalize(rows_by_tag, audit)
    save_audit(audit)
    counts = build_filtered_indexes(rows_by_tag, audit)
    rankings = retrieve(rows_by_tag)
    result = evaluate(rankings, counts, totals, audit)
    update_report(result)
    for tag in ("B", "E3", "D30"):
        print(
            f"[filter:{tag}] {counts[tag]}/{totals[tag]} "
            f"({result['retained_pct'][tag]:.1f}%)"
        )
    for key, label in (
        ("evidence_recall@5", "Evidence Recall@5"),
        ("mrr@10", "MRR@10"),
    ):
        print(
            f"  {label:<20} "
            + "  ".join(
                f"{tag}={result['metrics'][key][tag]['mean']:.3f}"
                for tag in (
                    "B",
                    "B-filtered",
                    "E3",
                    "E3-filtered",
                    "D30",
                    "D30-filtered",
                )
            )
        )
    return result


if __name__ == "__main__":
    run()
