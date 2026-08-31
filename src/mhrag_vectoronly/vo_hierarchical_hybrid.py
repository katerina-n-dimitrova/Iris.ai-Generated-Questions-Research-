"""
Hierarchical hybrid RAG experiment on the closed 15-article MultiHop-RAG set.

Configuration:
  * sentence/paragraph-aware chunks: 512 tokens, 128-token overlap
  * 10 verified document-level routing questions per complete article
  * document routing: BM25(question text) + dense(question embeddings) + RRF
  * chunk retrieval inside routed documents: BM25(chunk text) + dense + RRF
  * cross-encoder reranking, overlap deduplication, LLM evidence validation
  * grounded answer decision saved for every evaluation query

The retrieval metrics use the same overlap-aware gold evidence units and metric
definitions as the existing vector-only experiment.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi

import vo_config as C
import vo_data as D
import vo_metrics as VM
from embeddings import embedding_signature, get_embedder

if (C.CHUNK_SIZE, C.CHUNK_OVERLAP) != (512, 128):
    raise RuntimeError("Hierarchical experiment requires 512/128 configuration")

VM.KS = C.TOP_K_VALUES

TAG = "hierarchical_512_128"
DATA = C.DATA_DIR / TAG
RESULTS = C.RESULTS_DIR / TAG
DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

ARTICLES_PATH = DATA / "articles.jsonl"
CHUNKS_PATH = DATA / "chunks.jsonl"
QUERIES_PATH = DATA / "queries.jsonl"
GOLD_PATH = DATA / "gold.jsonl"
SUMMARY_PATH = DATA / "dataset_summary.json"
DOCQ_PATH = DATA / "verified_document_questions.jsonl"
QUERY_ANALYSIS_PATH = DATA / "query_understanding.jsonl"
RANKINGS_PATH = RESULTS / "rankings.json"
VALIDATION_PATH = RESULTS / "evidence_validation_answers.jsonl"
METRICS_PATH = RESULTS / "metrics.json"
REPORT = C.PROJECT_ROOT / "report" / "mhrag_15articles_hierarchical_512_128.html"

TEXT_COLL = "mhrag_vo15_hier512_128_chunks"
DOCQ_COLL = "mhrag_vo15_hier512_128_doc_questions"

ENC = tiktoken.get_encoding(C.TOKENIZER)
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*")
DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?",
    re.I,
)
NUMBER_RE = re.compile(r"(?<!\w)[<>≤≥]?\s*\$?\d+(?:\.\d+)?%?")
NEGATIONS = {
    "not",
    "without",
    "except",
    "exclude",
    "excluding",
    "before",
    "after",
    "above",
    "below",
    "between",
    "neither",
    "nor",
}
STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "by",
    "with",
    "and",
    "or",
    "that",
    "this",
    "these",
    "those",
    "which",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "does",
    "did",
    "do",
    "has",
    "have",
    "had",
    "as",
}
EXPANSIONS = {
    "cost": ["price"],
    "price": ["cost"],
    "purchase": ["buy"],
    "buy": ["purchase"],
    "physician": ["doctor"],
    "doctor": ["physician"],
    "launch": ["release"],
    "release": ["launch"],
}

DOC_ROUTE_K = 5
CANDIDATE_K = 20
RRF_K = 60


def read_jsonl(path: Path) -> List[dict]:
    return D.read_jsonl(path)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    D._write_jsonl(path, rows)


def tokens(text: str) -> List[str]:
    """BM25 tokenizer: preserve negations, names, numbers and date components."""
    return [x.casefold() for x in WORD_RE.findall(text)]


def _sentences(paragraph: str) -> List[str]:
    return [s.strip() for s in SENT_RE.split(paragraph) if s.strip()]


def _semantic_chunks(article: dict) -> List[dict]:
    """Pack complete sentences; overlap by trailing complete sentences."""
    units = []
    for pi, paragraph in enumerate(article["cleaned_body"].split("\n\n")):
        for sentence in _sentences(paragraph):
            stoks = ENC.encode(sentence)
            if len(stoks) <= C.CHUNK_SIZE:
                units.append((pi, sentence, len(stoks)))
            else:
                # Unavoidably split a single overlong sentence.
                for start in range(0, len(stoks), C.CHUNK_SIZE):
                    part = ENC.decode(stoks[start : start + C.CHUNK_SIZE]).strip()
                    units.append((pi, part, len(ENC.encode(part))))

    windows, current, current_n = [], [], 0
    i = 0
    while i < len(units):
        pi, text, n = units[i]
        if current and current_n + n > C.CHUNK_SIZE:
            windows.append(current)
            overlap, overlap_n = [], 0
            for u in reversed(current):
                overlap.insert(0, u)
                overlap_n += u[2]
                if overlap_n >= C.CHUNK_OVERLAP:
                    break
            # Prevent a single 512-token unit from blocking forward movement.
            current = overlap if len(overlap) < len(current) else []
            current_n = sum(u[2] for u in current)
            continue
        current.append((pi, text, n))
        current_n += n
        i += 1
    if current and (not windows or current != windows[-1]):
        windows.append(current)

    chunks = []
    for ci, window in enumerate(windows):
        paragraph_ids = sorted({u[0] for u in window})
        content = " ".join(u[1] for u in window).strip()
        chunks.append(
            {
                "document_id": article["article_id"],
                "chunk_id": f"{article['article_key']}::h{ci}",
                "document_title": article["title"],
                "section_title": None,
                "subsection_title": None,
                "source": article["source"],
                "date": article["published_at"],
                "authors": [],
                "entities": [],
                "topics": [],
                "paragraph_ids": paragraph_ids,
                "chunk_position": ci,
                "n_tokens": len(ENC.encode(content)),
                "content": content,
            }
        )
    return chunks


def prepare(force: bool = False) -> dict:
    if SUMMARY_PATH.exists() and not force:
        return json.load(SUMMARY_PATH.open())

    # Reuse the exact selected articles, but rebuild 512/128 semantic chunks.
    source_articles = C.DATA_DIR / "processed_articles_512_256.jsonl"
    if not source_articles.exists():
        source_articles = C.PROCESSED_ARTICLES
    articles = read_jsonl(source_articles)
    selected_ids = {a["article_id"] for a in articles}
    chunks = [c for a in articles for c in _semantic_chunks(a)]
    chunks_by_doc = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c["document_id"]].append(c)

    raw_queries = D.load_all_queries()
    queries, gold_rows, unresolved = [], [], []
    for q in raw_queries:
        if q["question_type"] == "null_query" or not q.get("evidence_list"):
            continue
        req = {e[C.ARTICLE_ID_FIELD] for e in q["evidence_list"]}
        if not req or not req <= selected_ids:
            continue
        facts, bad = [], False
        for ei, evidence in enumerate(q["evidence_list"]):
            fact = evidence.get("fact", "")
            hits, score, method = D._locate_fact(
                fact,
                [
                    {"chunk_id": c["chunk_id"], "text": c["content"]}
                    for c in chunks_by_doc[evidence[C.ARTICLE_ID_FIELD]]
                ],
            )
            if not hits:
                unresolved.append(
                    {
                        "query_id": q["query_id"],
                        "fact": fact,
                        "best_score": score,
                    }
                )
                bad = True
            facts.append(
                {
                    "evidence_fact_id": f"{q['query_id']}::e{ei}",
                    "fact": fact,
                    "chunk_ids": hits,
                    "match": method,
                }
            )
        if bad:
            continue
        units = [sorted(set(x["chunk_ids"])) for x in facts]
        gold_ids = sorted({cid for unit in units for cid in unit})
        row = {
            "query_id": q["query_id"],
            "query": q["query"].strip(),
            "question_type": q["question_type"].replace("_query", ""),
            "gold_answer": q.get("answer", ""),
            "required_article_ids": sorted(req),
            "n_required_documents": len(req),
            "n_required_evidence_facts": len(facts),
            "gold_chunk_ids": gold_ids,
            "evidence_units": units,
        }
        queries.append(row)
        gold_rows.append(
            {
                "query_id": q["query_id"],
                "facts": facts,
                "gold_chunk_ids": gold_ids,
                "evidence_units": units,
            }
        )
    write_jsonl(ARTICLES_PATH, articles)
    write_jsonl(CHUNKS_PATH, chunks)
    write_jsonl(QUERIES_PATH, queries)
    write_jsonl(GOLD_PATH, gold_rows)
    summary = {
        "articles": len(articles),
        "chunks": len(chunks),
        "queries": len(queries),
        "unresolved_facts": len(unresolved),
        "chunk_size": 512,
        "chunk_overlap": 128,
        "avg_chunk_tokens": float(np.mean([c["n_tokens"] for c in chunks])),
        "sentence_boundary_preserved": True,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print("[hier:data]", summary)
    return summary


def load_chunks() -> List[dict]:
    return read_jsonl(CHUNKS_PATH)


def load_queries() -> List[dict]:
    return read_jsonl(QUERIES_PATH)


def load_gold() -> Dict[str, dict]:
    return {x["query_id"]: x for x in read_jsonl(GOLD_PATH)}


GEN_SYSTEM = (
    "Generate document-routing questions from a complete news document. "
    "Questions must be explicit, discriminative, diverse, and grounded. "
    "Every evidence quote must be copied verbatim. Output JSON only."
)
VERIFY_SYSTEM = (
    "Strictly verify document-routing questions against the complete document "
    "and cited chunks. Do not rewrite questions. Output JSON only."
)


def _map_evidence(evidence: str, chunks: List[dict]) -> List[str]:
    if not evidence:
        return []
    exact = [c["chunk_id"] for c in chunks if evidence in c["content"]]
    if exact:
        return exact

    # Recover harmless quote/whitespace normalization. The separate verifier
    # still decides whether the cited chunk actually supports the candidate.
    normalize = lambda text: re.sub(r"\W+", " ", text.casefold()).strip()
    target = normalize(evidence)
    normalized = [(c["chunk_id"], normalize(c["content"])) for c in chunks]
    contained = [
        cid for cid, text in normalized if target and (target in text or text in target)
    ]
    if contained:
        return contained
    target_terms = set(target.split())
    scored = []
    for cid, text in normalized:
        text_terms = set(text.split())
        coverage = (
            len(target_terms & text_terms) / len(target_terms) if target_terms else 0.0
        )
        sentence_similarity = max(
            (
                SequenceMatcher(None, target, normalize(sentence)).ratio()
                for sentence in re.split(
                    r"(?<=[.!?])\s+",
                    next(c["content"] for c in chunks if c["chunk_id"] == cid),
                )
            ),
            default=0.0,
        )
        scored.append((max(coverage, sentence_similarity), cid))
    scored.sort(reverse=True)
    return [scored[0][1]] if scored and scored[0][0] >= 0.72 else []


def _gen_call(client, article: dict, count: int, existing: List[dict]) -> dict:
    prior = "\n".join(f"- {x['question']}" for x in existing)
    prompt = f'''Complete document:
"""
{article["cleaned_body"]}
"""

Generate exactly {count} additional diverse document-level retrieval questions.
Existing questions that must not be repeated:
{prior or "(none)"}

Cover distinct combinations of main topics, entities, products, organizations,
people, methods, comparisons, numerical results, applications, limitations,
conclusions, dates, locations, and information needs when present. Include
factual, comparison, process, numerical/result, limitation, and application
intents. Do not combine unrelated sections.

Every question must be explicitly answerable from the document, specific enough
to identify this document, and free of unsupported assumptions. Return a concise
short answer and one short VERBATIM supporting quote.

JSON:
{{"questions":[{{"question":"...","short_answer":"...","evidence":"...",
"question_type":"factual|comparison|process|numerical|limitation|application",
"important_entities":["..."]}}, ...]}}'''
    response = client.chat.completions.create(
        model=C.gen_model(),
        temperature=C.GEN_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


def _verify_call(
    client, article: dict, chunks: List[dict], candidates: List[dict]
) -> List[bool]:
    cited = sorted({cid for x in candidates for cid in x["supporting_chunk_ids"]})
    chunk_map = {c["chunk_id"]: c for c in chunks}
    chunk_text = "\n\n".join(f"[{cid}]\n{chunk_map[cid]['content']}" for cid in cited)
    payload = [
        {
            "id": i,
            "question": x["question"],
            "short_answer": x["short_answer"],
            "supporting_chunk_ids": x["supporting_chunk_ids"],
            "question_type": x["question_type"],
            "important_entities": x["important_entities"],
        }
        for i, x in enumerate(candidates)
    ]
    prompt = f'''Complete document:
"""
{article["cleaned_body"]}
"""

Cited supporting chunks:
{chunk_text}

Candidates:
{json.dumps(payload, ensure_ascii=False)}

For each candidate, set pass=true only if ALL are true: explicitly answerable;
answer traceable to cited chunks; chunk IDs correct; specific and useful for
identifying this document; meaningfully distinct; no unrelated-section merge;
answer fully supported; distinguishing entities included when necessary; no
hallucination or outside assumption.

Return JSON: {{"verdicts":[{{"id":0,"pass":true,"reasons":[]}}, ...]}}'''
    response = client.chat.completions.create(
        model=C.gen_model(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    data = json.loads(response.choices[0].message.content)
    verdicts = {int(x["id"]): bool(x.get("pass")) for x in data.get("verdicts", [])}
    return [verdicts.get(i, False) for i in range(len(candidates))]


def _generate_document(client, article: dict, chunks: List[dict]) -> dict:
    accepted, seen = [], set()
    for _ in range(5):
        if len(accepted) >= 10:
            break
        need = 10 - len(accepted)
        try:
            raw = _gen_call(client, article, need, accepted)
            candidates = []
            for item in raw.get("questions", []):
                q = str(item.get("question", "")).strip()
                evidence = str(item.get("evidence", "")).strip()
                key = re.sub(r"\W+", " ", q.casefold()).strip()
                supporting = _map_evidence(evidence, chunks)
                if not q or key in seen or not supporting:
                    continue
                candidates.append(
                    {
                        "question": q,
                        "short_answer": str(item.get("short_answer", "")).strip(),
                        "evidence": evidence,
                        "supporting_chunk_ids": supporting,
                        "question_type": str(item.get("question_type", "factual")),
                        "important_entities": [
                            str(x) for x in item.get("important_entities", [])
                        ],
                        "document_id": article["article_id"],
                    }
                )
            if not candidates:
                continue
            verdicts = _verify_call(client, article, chunks, candidates)
            for candidate, passed in zip(candidates, verdicts):
                if passed and len(accepted) < 10:
                    candidate["question_id"] = (
                        f"{article['article_key']}::dq{len(accepted)}"
                    )
                    candidate["verified"] = True
                    accepted.append(candidate)
                    seen.add(
                        re.sub(r"\W+", " ", candidate["question"].casefold()).strip()
                    )
        except Exception as exc:
            print(
                f"[hier:docq:error] {article['article_key']}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
    return {
        "document_id": article["article_id"],
        "document_title": article["title"],
        "questions": accepted,
        "valid": len(accepted) == 10,
    }


def generate_document_questions(force: bool = False) -> List[dict]:
    articles, chunks = read_jsonl(ARTICLES_PATH), load_chunks()
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["document_id"]].append(c)
    cache = (
        {x["document_id"]: x for x in read_jsonl(DOCQ_PATH)}
        if DOCQ_PATH.exists() and not force
        else {}
    )
    todo = [a for a in articles if not cache.get(a["article_id"], {}).get("valid")]
    print(f"[hier:docq] to-do {len(todo)}/15")
    if todo:
        client = C.openai_client()
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                ex.submit(_generate_document, client, a, by_doc[a["article_id"]]): a
                for a in todo
            }
            for fut in as_completed(futures):
                a = futures[fut]
                cache[a["article_id"]] = fut.result()
    rows = [cache[a["article_id"]] for a in articles]
    write_jsonl(DOCQ_PATH, rows)
    print(
        f"[hier:docq] verified {sum(len(x['questions']) for x in rows)} "
        f"| valid docs {sum(x['valid'] for x in rows)}/15"
    )
    return rows


def analyze_query(q: dict) -> dict:
    original = q["query"]
    raw = tokens(original)
    keywords = [x for x in raw if x not in STOP or x in NEGATIONS]
    keyphrases = [
        " ".join(m.group(0).split())
        for m in re.finditer(
            r"\b(?:[A-Z][\w.-]*)(?:\s+(?:[A-Z][\w.-]*|of|the|and)){0,4}",
            original,
        )
    ]
    entities = {f"entity_{i}": x for i, x in enumerate(keyphrases)}
    dates = DATE_RE.findall(original)
    numbers = NUMBER_RE.findall(original)
    negations = [x for x in raw if x in NEGATIONS]
    expanded = {x: EXPANSIONS[x] for x in keywords if x in EXPANSIONS}
    comparison = any(
        x in raw for x in ("compare", "versus", "vs", "than", "difference", "both")
    )
    return {
        "query_id": q["query_id"],
        "original_query": original,
        "intent": (
            "comparison"
            if comparison
            else "temporal"
            if dates or "when" in raw
            else "fact_retrieval"
        ),
        "keywords": keywords,
        "keyphrases": keyphrases,
        "entities": entities,
        "dates": dates,
        "locations": [],
        "numeric_constraints": {"values": numbers} if numbers else {},
        "inclusion_constraints": {},
        "exclusion_constraints": {"terms": negations} if negations else {},
        "negations": negations,
        "metadata_filters": {},
        "expanded_terms": expanded,
        "requires_clarification": False,
        "clarification_reason": None,
    }


def query_understanding(force: bool = False) -> List[dict]:
    if QUERY_ANALYSIS_PATH.exists() and not force:
        return read_jsonl(QUERY_ANALYSIS_PATH)
    rows = [analyze_query(q) for q in load_queries()]
    write_jsonl(QUERY_ANALYSIS_PATH, rows)
    return rows


@dataclass
class Indexes:
    chunks: List[dict]
    chunk_map: Dict[str, dict]
    chunk_vectors: np.ndarray
    chunk_bm25: BM25Okapi
    doc_questions: List[dict]
    docq_vectors: np.ndarray
    docq_bm25: BM25Okapi


def build_indexes(doc_rows: List[dict]) -> Indexes:
    embedder = get_embedder()
    chunks = load_chunks()
    chunk_vectors = np.asarray(
        embedder.embed_documents([c["content"] for c in chunks]), dtype=float
    )
    chunk_coll = C.reset_collection(TEXT_COLL)
    chunk_coll.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=chunk_vectors.tolist(),
        documents=[c["content"] for c in chunks],
        metadatas=[
            {
                "parent_chunk_id": c["chunk_id"],
                "parent_document_id": c["document_id"],
                "document_title": c["document_title"],
                "source": c["source"],
                "date": c["date"],
            }
            for c in chunks
        ],
    )
    questions = [q for row in doc_rows for q in row["questions"]]
    if len(questions) != 150:
        raise RuntimeError("Need exactly 150 verified document questions")
    docq_vectors = np.asarray(
        embedder.embed_documents([q["question"] for q in questions]), dtype=float
    )
    doc_coll = C.reset_collection(DOCQ_COLL)
    doc_coll.add(
        ids=[q["question_id"] for q in questions],
        embeddings=docq_vectors.tolist(),
        documents=[q["question"] for q in questions],
        metadatas=[
            {
                "question_id": q["question_id"],
                "document_id": q["document_id"],
                "short_answer": q["short_answer"],
                "supporting_chunk_ids": json.dumps(q["supporting_chunk_ids"]),
                "question_type": q["question_type"],
                "important_entities": json.dumps(q["important_entities"]),
            }
            for q in questions
        ],
    )
    print(f"[hier:index] chunks={len(chunks)} doc_questions={len(questions)}")
    return Indexes(
        chunks=chunks,
        chunk_map={c["chunk_id"]: c for c in chunks},
        chunk_vectors=chunk_vectors,
        chunk_bm25=BM25Okapi([tokens(c["content"]) for c in chunks]),
        doc_questions=questions,
        docq_vectors=docq_vectors,
        docq_bm25=BM25Okapi([tokens(q["question"]) for q in questions]),
    )


def _cosine_matrix(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    # Iris may return low-precision or unusually scaled vectors. Normalize
    # each operand by its largest absolute coordinate before norm/dot
    # accumulation; this preserves cosine direction and prevents overflow.
    matrix = np.asarray(matrix, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    matrix[~np.isfinite(matrix)] = 0.0
    vector[~np.isfinite(vector)] = 0.0
    row_scale = np.max(np.abs(matrix), axis=1, keepdims=True)
    row_scale[row_scale == 0] = 1.0
    vector_scale = float(np.max(np.abs(vector)))
    if vector_scale == 0:
        vector_scale = 1.0
    matrix = matrix / row_scale
    vector = vector / vector_scale
    mn = np.linalg.norm(matrix, axis=1)
    vn = np.linalg.norm(vector)
    denom = mn * vn
    dots = np.einsum("ij,j->i", matrix, vector, dtype=np.float64)
    scores = np.full(len(matrix), -1.0, dtype=np.float64)
    np.divide(dots, denom, out=scores, where=denom > 1e-12)
    return np.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)


def _rank_indices(
    scores: Sequence[float], allowed: set[int] | None = None, k: int | None = None
) -> List[int]:
    ids = list(range(len(scores))) if allowed is None else list(allowed)
    ids.sort(key=lambda i: (-float(scores[i]), i))
    return ids[:k] if k else ids


def _rrf(rankings: Sequence[Sequence[str]]) -> Dict[str, float]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1.0 / (RRF_K + rank)
    return dict(scores)


def _bm25_query_terms(analysis: dict) -> List[str]:
    combined = (
        tokens(analysis["original_query"])
        + analysis["keywords"]
        + [x.casefold() for p in analysis["keyphrases"] for x in tokens(p)]
        + [x.casefold() for x in analysis["entities"].values()]
        + [y for values in analysis["expanded_terms"].values() for y in values]
    )
    return combined


def route_documents(
    index: Indexes, qvec: np.ndarray, analysis: dict
) -> tuple[List[str], dict]:
    bm = index.docq_bm25.get_scores(_bm25_query_terms(analysis))
    dense = _cosine_matrix(index.docq_vectors, qvec)
    bm_order = _rank_indices(bm)
    dense_order = _rank_indices(dense)

    def doc_ranking(order):
        seen, out = set(), []
        for i in order:
            did = index.doc_questions[i]["document_id"]
            if did not in seen:
                seen.add(did)
                out.append(did)
        return out

    bm_docs, dense_docs = doc_ranking(bm_order), doc_ranking(dense_order)
    fused = _rrf([bm_docs, dense_docs])
    routed = sorted(fused, key=lambda d: (-fused[d], d))[:DOC_ROUTE_K]
    return routed, {
        "bm25_documents": bm_docs,
        "vector_documents": dense_docs,
        "rrf_scores": fused,
    }


def _chunk_hybrid(
    index: Indexes, qvec: np.ndarray, analysis: dict, allowed_docs: set[str] | None
) -> tuple[List[str], dict]:
    allowed = {
        i
        for i, c in enumerate(index.chunks)
        if allowed_docs is None or c["document_id"] in allowed_docs
    }
    bm = index.chunk_bm25.get_scores(_bm25_query_terms(analysis))
    dense = _cosine_matrix(index.chunk_vectors, qvec)
    bm_ids = _rank_indices(bm, allowed, CANDIDATE_K)
    dense_ids = _rank_indices(dense, allowed, CANDIDATE_K)
    bm_chunks = [index.chunks[i]["chunk_id"] for i in bm_ids]
    dense_chunks = [index.chunks[i]["chunk_id"] for i in dense_ids]
    fused = _rrf([bm_chunks, dense_chunks])
    ranking = sorted(fused, key=lambda cid: (-fused[cid], cid))
    return ranking, {
        "bm25": bm_chunks,
        "vector": dense_chunks,
        "rrf_scores": fused,
    }


def _dedup(ranked: List[str], chunk_map: Dict[str, dict], limit: int = 10) -> List[str]:
    kept, token_sets = [], []
    for cid in ranked:
        ts = set(tokens(chunk_map[cid]["content"]))
        duplicate = False
        for old_cid, old in zip(kept, token_sets):
            if chunk_map[old_cid]["document_id"] != chunk_map[cid]["document_id"]:
                continue
            union = len(ts | old)
            if union and len(ts & old) / union >= 0.82:
                duplicate = True
                break
        if not duplicate:
            kept.append(cid)
            token_sets.append(ts)
        if len(kept) >= limit:
            break
    return kept


def _as_metric_row(
    q: dict, gold: dict, cids: List[str], chunk_map: Dict[str, dict]
) -> dict:
    return {
        "query_id": q["query_id"],
        "question_type": q["question_type"],
        "n_required_documents": q["n_required_documents"],
        "n_required_evidence_facts": q["n_required_evidence_facts"],
        "gold_chunk_ids": gold["gold_chunk_ids"],
        "evidence_units": gold["evidence_units"],
        "required_article_ids": q["required_article_ids"],
        "ranked": [
            {
                "rank": i,
                "chunk_id": cid,
                "parent_document_id": chunk_map[cid]["document_id"],
                "score": float(len(cids) - i + 1),
            }
            for i, cid in enumerate(cids, 1)
        ],
    }


VALIDATE_SYSTEM = (
    "Validate retrieved evidence and produce a grounded answer decision. Use "
    "only supplied chunks. Output JSON only."
)


def _validate_and_answer(
    client, q: dict, candidates: List[str], chunk_map: Dict[str, dict]
) -> dict:
    evidence = "\n\n".join(
        f"[{cid}] {chunk_map[cid]['document_title']}\n{chunk_map[cid]['content']}"
        for cid in candidates
    )
    prompt = f"""User query:
{q["query"]}

Candidate chunks:
{evidence}

For each chunk label it directly_supports, partially_supports,
related_but_insufficient, contradicts, or irrelevant. Check correct entities,
values, constraints, negations, currency/units, dates, location, missing
conditions, and conflicts.

Then choose response_mode: strong_evidence, multiple_valid_answers,
ambiguous_query, or insufficient_evidence. Generate an answer using only chunks
labeled directly_supports or partially_supports. Cite chunk IDs. If ambiguous,
ask a focused clarification or give a qualified comparison. Never invent.

JSON:
{{"chunks":[{{"chunk_id":"...","label":"...","reason":"..."}}],
"response_mode":"...","answer":"...","used_chunk_ids":["..."]}}"""
    response = client.chat.completions.create(
        model=C.gen_model(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VALIDATE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


def retrieve_all(
    index: Indexes, analyses: List[dict], force_validation: bool = False
) -> Dict[str, List[dict]]:
    queries, gold = load_queries(), load_gold()
    analysis_by_id = {x["query_id"]: x for x in analyses}
    embedder = get_embedder()
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    validation_cache = (
        {x["query_id"]: x for x in read_jsonl(VALIDATION_PATH)}
        if VALIDATION_PATH.exists() and not force_validation
        else {}
    )
    conditions = {
        x: []
        for x in ("A-vector", "Flat-hybrid", "Hier-hybrid", "Hier-CE", "Hier-final")
    }
    route_debug = {}
    todo_validation = []

    for q in queries:
        qvec = np.asarray(embedder.embed_query(q["query"]), dtype=float)
        analysis = analysis_by_id[q["query_id"]]
        g = gold[q["query_id"]]

        dense = _cosine_matrix(index.chunk_vectors, qvec)
        a_ids = [
            index.chunks[i]["chunk_id"] for i in _rank_indices(dense, k=C.RANK_DEPTH)
        ]
        conditions["A-vector"].append(_as_metric_row(q, g, a_ids, index.chunk_map))

        flat, _ = _chunk_hybrid(index, qvec, analysis, None)
        conditions["Flat-hybrid"].append(
            _as_metric_row(q, g, flat[: C.RANK_DEPTH], index.chunk_map)
        )

        routed_docs, doc_debug = route_documents(index, qvec, analysis)
        hierarchical, chunk_debug = _chunk_hybrid(
            index, qvec, analysis, set(routed_docs)
        )
        conditions["Hier-hybrid"].append(
            _as_metric_row(q, g, hierarchical[: C.RANK_DEPTH], index.chunk_map)
        )

        ce_candidates = hierarchical[:CANDIDATE_K]
        ce_scores = reranker.predict(
            [(q["query"], index.chunk_map[cid]["content"]) for cid in ce_candidates],
            batch_size=32,
            show_progress_bar=False,
        )
        ce_ranked = [
            cid
            for cid, _ in sorted(
                zip(ce_candidates, ce_scores),
                key=lambda x: (-float(x[1]), x[0]),
            )
        ]
        deduped = _dedup(ce_ranked, index.chunk_map, C.RANK_DEPTH)
        conditions["Hier-CE"].append(_as_metric_row(q, g, deduped, index.chunk_map))

        route_debug[q["query_id"]] = {
            "selected_documents": routed_docs,
            "document_routing": doc_debug,
            "chunk_retrieval": chunk_debug,
            "cross_encoder_ranking": ce_ranked,
        }
        if q["query_id"] not in validation_cache:
            todo_validation.append((q, deduped))

    if todo_validation:
        print(f"[hier:validate] to-do {len(todo_validation)}")
        client = C.openai_client()
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {
                ex.submit(_validate_and_answer, client, q, cids, index.chunk_map): (
                    q,
                    cids,
                )
                for q, cids in todo_validation
            }
            for fut in as_completed(futures):
                q, cids = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        "chunks": [],
                        "response_mode": "insufficient_evidence",
                        "answer": "Evidence validation failed.",
                        "used_chunk_ids": [],
                        "error": str(exc),
                    }
                validation_cache[q["query_id"]] = {
                    "query_id": q["query_id"],
                    "query": q["query"],
                    "candidate_chunk_ids": cids,
                    **result,
                }
        write_jsonl(
            VALIDATION_PATH,
            [validation_cache[q["query_id"]] for q in queries],
        )

    for q in queries:
        g = gold[q["query_id"]]
        validated = validation_cache[q["query_id"]]
        labels = {
            x.get("chunk_id"): x.get("label") for x in validated.get("chunks", [])
        }
        cids = [
            cid
            for cid in validated["candidate_chunk_ids"]
            if labels.get(cid) in ("directly_supports", "partially_supports")
        ]
        conditions["Hier-final"].append(_as_metric_row(q, g, cids, index.chunk_map))

    RANKINGS_PATH.write_text(
        json.dumps(
            {
                "conditions": conditions,
                "routing_debug": route_debug,
            }
        )
    )
    return conditions


def evaluate(conditions: Dict[str, List[dict]], summary: dict) -> dict:
    per = {
        tag: {r["query_id"]: VM.per_query(r) for r in rows}
        for tag, rows in conditions.items()
    }
    qids = sorted(per["A-vector"])
    metric_defs = [
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ]

    def vals(tag, key):
        return np.array([per[tag][qid][key] for qid in qids])

    metrics = {}
    for key, _ in metric_defs:
        base = vals("A-vector", key)
        metrics[key] = {}
        for tag in per:
            x = vals(tag, key)
            lo, hi = _ci(x)
            item = {"mean": float(x.mean()), "ci_low": lo, "ci_high": hi}
            if tag != "A-vector":
                dm, dlo, dhi, sig = _dci(base, x)
                item.update(
                    delta=float(dm),
                    delta_low=dlo,
                    delta_high=dhi,
                    significant=bool(sig),
                )
            metrics[key][tag] = item
    result = {
        "dataset": summary,
        "n_queries": len(qids),
        "embedding_signature": embedding_signature(),
        "document_route_k": DOC_ROUTE_K,
        "candidate_k_per_retriever": CANDIDATE_K,
        "rrf_k": RRF_K,
        "metrics": metrics,
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2))
    return result


def _ci(x, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    means = x[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _dci(base, arm, n=1000, seed=42):
    diff = arm - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return float(diff.mean()), lo, hi, bool(lo > 0 or hi < 0)


def _cell(item: dict, base: bool) -> str:
    if base:
        return (
            f'<div class="v">{item["mean"]:.3f}</div>'
            f'<div class="ci">[{item["ci_low"]:.3f}, '
            f"{item['ci_high']:.3f}]</div>"
        )
    d = item["delta"]
    cls = "good" if d > 1e-9 else ("bad" if d < -1e-9 else "flat")
    star = "<b>*</b>" if item["significant"] else ""
    return (
        f'<div class="v">{item["mean"]:.3f}</div>'
        f'<div class="d {cls}" title="Δ 95% CI '
        f'[{item["delta_low"]:+.3f}, {item["delta_high"]:+.3f}]">'
        f"({d:+.3f}{star})</div>"
    )


def render(result: dict) -> None:
    metric_defs = [
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ]
    labels = {
        "A-vector": "Original chunks · vector-only baseline",
        "Flat-hybrid": "All chunks · BM25 + vector RRF",
        "Hier-hybrid": "Document routing → restricted chunk hybrid",
        "Hier-CE": "Hierarchical hybrid + cross-encoder + dedup",
        "Hier-final": "Hierarchical + validated supporting evidence",
        "GEPA-Hier": "GEPA prompt · document routing → restricted chunk hybrid",
        "GEPA-CE": "GEPA prompt · hierarchical + cross-encoder + dedup",
        "GEPA-final": "GEPA prompt · validated supporting evidence",
    }
    body = ""
    for tag in (tag for tag in labels if tag in result["metrics"]["evidence_recall@1"]):
        cells = "".join(
            f"<td>{_cell(result['metrics'][key][tag], tag == 'A-vector')}</td>"
            for key, _ in metric_defs
        )
        body += (
            f'<tr class="{"base" if tag == "A-vector" else ""}">'
            f'<td class="nm"><b>{escape(tag)}</b> · {escape(labels[tag])}</td>'
            f"{cells}</tr>"
        )
    heads = "".join(f"<th>{label}</th>" for _, label in metric_defs)
    s = result["dataset"]
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MultiHop-RAG hierarchical hybrid 512/128</title><style>
:root{{--bg:#fff;--fg:#161a21;--muted:#5b6572;--line:#e3e7ee;--card:#f7f9fc;--good:#0a7d3f;--bad:#c0392b;--flat:#8792a1}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1319;--fg:#e6eaf0;--muted:#9aa5b3;--line:#242b36;--card:#161c25;--good:#38d17a;--bad:#ff6b5e;--flat:#8792a1}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.wrap{{max-width:1250px;margin:auto;padding:34px 22px 70px}}h1{{font-size:24px;margin:0 0 5px}}h2{{font-size:18px;margin:30px 0 8px}}.cap{{color:var(--muted);font-size:12.5px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid var(--line);padding:9px;text-align:center}}th{{background:var(--card)}}td.nm,th:first-child{{text-align:left}}tr.base{{background:var(--card)}}.v{{font-weight:700;font-size:14px}}.ci,.d{{font-size:11px}}.ci,.flat{{color:var(--muted)}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}
</style></head><body><div class="wrap">
<h1>15-article MultiHop-RAG — hierarchical hybrid RAG — 512/128</h1>
<p class="cap">Document-level routing with 150 verified generated questions:
BM25 + Iris vector + RRF. Restricted chunk retrieval: original-text BM25 +
Iris chunk vectors + RRF, followed by cross-encoder reranking, overlap
deduplication, and evidence validation.</p>
<table><thead><tr><th>Condition</th>{heads}</tr></thead><tbody>{body}</tbody></table>
<p class="cap">n={result["n_queries"]} queries · {s["articles"]} articles ·
{s["chunks"]} sentence-aware chunks · target 512 tokens / 128 overlap ·
{escape(result["embedding_signature"])} · top {DOC_ROUTE_K} routed documents ·
top {CANDIDATE_K} candidates/retriever · RRF k={RRF_K}. Deltas are versus
A-vector; * means paired 95% bootstrap CI excludes zero.</p>
<h2>Pipeline integrity</h2>
<p class="cap">Full original queries are retained for dense search and included
in BM25 alongside conservative extracted terms. BM25 uses tokenized text only,
never embeddings. Generated questions route documents but do not replace chunk
evidence retrieval. Grounded answers and per-chunk validation labels are saved
to <code>{escape(str(VALIDATION_PATH))}</code>.</p>
</div></body></html>"""
    REPORT.write_text(doc)
    print(f"[hier:report] {REPORT}")


def run() -> dict:
    summary = prepare(force=False)
    doc_rows = generate_document_questions(force=False)
    if not all(x["valid"] and len(x["questions"]) == 10 for x in doc_rows):
        raise RuntimeError("Document question generation incomplete; rerun")
    analyses = query_understanding(force=False)
    index = build_indexes(doc_rows)
    conditions = retrieve_all(index, analyses, force_validation=False)
    result = evaluate(conditions, summary)
    render(result)
    for key, label in (
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ):
        print(
            f"  {label:<20} "
            + "  ".join(
                f"{tag}={result['metrics'][key][tag]['mean']:.3f}"
                for tag in (
                    "A-vector",
                    "Flat-hybrid",
                    "Hier-hybrid",
                    "Hier-CE",
                    "Hier-final",
                )
            )
        )
    return result


if __name__ == "__main__":
    run()
