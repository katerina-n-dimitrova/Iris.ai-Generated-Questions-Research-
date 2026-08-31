"""
Synthetic Q&A EVALUATION set for the 15 DocBank documents.

For each document we show the LLM that document's chunks (id + layout type +
text) and ask for specific, unambiguous questions each answerable from ONE
chunk, tagged with the gold chunk id, verbatim evidence, a question type, and the
source/layout type. This set is the RETRIEVAL EVAL (queries + gold), kept
strictly separate from the doc2query enrichment questions (different prompt,
different file, never embedded into the index).

Outputs: results/docbank/docbank_15docs_eval_qa.{json,csv}
"""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import docbank_config as C
import config as base_config

QTYPES = [
    "section_title",
    "paragraph_fact",
    "table",
    "equation",
    "caption_reference",
    "method_result",
]
STYPES = ["paragraph", "table", "equation", "caption", "section", "mixed"]


def _prompt(doc_id: str, chunks: List[Dict], n: int):
    listing = []
    for c in chunks:
        listing.append(
            f"[{c['chunk_id']}] (type={c['layout_type']}, page={c['page']}) "
            f"{c['text'][:600]}"
        )
    body = "\n\n".join(listing)
    system = (
        "You build a retrieval EVALUATION set for one scientific document. You are "
        "given the document's chunks, each with an ID, layout type and text. Write "
        "specific, unambiguous questions that are each answerable from exactly ONE "
        "chunk. Rules: (1) the answer must be fully supported by that chunk — never "
        "use outside knowledge; (2) no vague questions (avoid 'what is discussed'); "
        "(3) name the specific entity/method/quantity/equation/table involved; "
        "(4) cover DIFFERENT chunks and a mix of question types; include table, "
        "equation and caption questions WHEN such chunks exist. "
        f"question_type in {QTYPES}; source_type in {STYPES} (match the gold chunk's "
        "layout type). Return ONLY a JSON object "
        '{"questions":[{"question","answer","gold_chunk_id","evidence",'
        '"question_type","source_type"}]}.'
    )
    user = (
        f"Document: {doc_id}\nChunks:\n{body}\n\n"
        f"Write {n} evaluation questions following the rules. gold_chunk_id MUST "
        "be one of the chunk IDs above. evidence MUST be a verbatim span copied "
        "from the gold chunk."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _gen_for_doc(client, doc_id: str, chunks: List[Dict], n: int):
    valid_ids = {c["chunk_id"] for c in chunks}
    by_id = {c["chunk_id"]: c for c in chunks}
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=C.LLM_MODEL,
        messages=_prompt(doc_id, chunks, n),
        temperature=C.LLM_TEMPERATURE,
        max_tokens=2200,
        response_format={"type": "json_object"},
    )
    secs = time.perf_counter() - t0
    u = resp.usage
    usage = (
        getattr(u, "prompt_tokens", 0) or 0,
        getattr(u, "completion_tokens", 0) or 0,
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
        items = data.get("questions", []) if isinstance(data, dict) else []
    except Exception:
        items = []
    out = []
    for it in items:
        gid = str(it.get("gold_chunk_id", "")).strip()
        q = str(it.get("question", "")).strip()
        if gid not in valid_ids or not q:
            continue  # must map to a real chunk
        st = str(it.get("source_type") or by_id[gid]["layout_type"])
        out.append(
            {
                "document_id": doc_id,
                "arxiv_id": by_id[gid]["arxiv_id"],
                "question": q,
                "answer": str(it.get("answer", "")).strip(),
                "gold_chunk_id": gid,
                "gold_evidence_text": str(it.get("evidence", "")).strip(),
                "question_type": str(it.get("question_type") or "paragraph_fact"),
                "source_type": st,
            }
        )
    return out, usage, secs


def generate_eval_qa(
    docs_chunks_by_doc: Dict[str, List[Dict]],
    *,
    n: int = C.QA_PER_DOC,
    workers: int = C.LLM_WORKERS,
) -> Dict:
    client = base_config.get_openai_client()
    rows: List[Dict] = []
    pt = ct = 0
    t_wall = time.perf_counter()
    gen_secs = 0.0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_gen_for_doc, client, d, cs, n): d
            for d, cs in docs_chunks_by_doc.items()
        }
        for f in as_completed(futs):
            out, usage, secs = f.result()
            rows.extend(out)
            pt += usage[0]
            ct += usage[1]
            gen_secs += secs
    # assign question ids
    for i, r in enumerate(rows):
        r["question_id"] = f"dbq_{i:04d}"
    rows.sort(key=lambda r: (r["document_id"], r["question_id"]))
    summary = {
        "num_qa": len(rows),
        "num_documents": len(docs_chunks_by_doc),
        "avg_qa_per_doc": round(len(rows) / max(len(docs_chunks_by_doc), 1), 1),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "gen_seconds_sum": round(gen_secs, 1),
        "wall_seconds": round(time.perf_counter() - t_wall, 1),
        "estimated_cost_usd": _cost(pt, ct),
    }
    return {"rows": rows, "summary": summary}


def _cost(pt, ct, model=C.LLM_MODEL):
    p = C.PRICE_PER_1M.get(model)
    return round(pt / 1e6 * p["input"] + ct / 1e6 * p["output"], 4) if p else -1.0


def save_eval_qa(rows: List[Dict]) -> Dict[str, Path]:
    jpath = C.RESULTS_DIR / "docbank_15docs_eval_qa.json"
    cpath = C.RESULTS_DIR / "docbank_15docs_eval_qa.csv"
    json.dump(rows, jpath.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cols = [
        "document_id",
        "question_id",
        "question",
        "answer",
        "gold_chunk_id",
        "gold_evidence_text",
        "question_type",
        "source_type",
    ]
    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return {"json": jpath, "csv": cpath}


def load_eval_qa() -> List[Dict]:
    jpath = C.RESULTS_DIR / "docbank_15docs_eval_qa.json"
    return json.load(jpath.open(encoding="utf-8")) if jpath.exists() else []
