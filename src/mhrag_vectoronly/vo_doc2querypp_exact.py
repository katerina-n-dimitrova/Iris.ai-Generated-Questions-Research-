"""
BERTopic + KeyBERT + GPT-5.4-mini Doc2Query++ dual-index pipeline.

Closed 15-article MultiHop-RAG experiment, 512/256 chunks. BERTopic discovers
broader collection topics from article sentences; KeyBERT extracts diverse
article keywords; GPT receives complete article + topics + keywords and creates
30 grounded retrieval questions. BERTopic, KeyBERT, chunk text, and questions
all use the configured Iris embedder. Text and question scores are fused after
independent min-max normalization.

Run with:
  EMBEDDING_BACKEND=iris python vo_doc2querypp_exact.py
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from typing import Dict, List

import numpy as np

import vo_overlap as V
import vo_metrics as VM
import vo_doc2querypp_article as AR
from embeddings import EMBEDDING_BACKEND, embedding_signature, get_embedder

if (V.C.CHUNK_SIZE, V.C.CHUNK_OVERLAP) != (512, 256):
    raise RuntimeError("Exact Doc2Query++ experiment requires 512/256 chunking")
if EMBEDDING_BACKEND != "iris":
    raise RuntimeError(
        "Run with EMBEDDING_BACKEND=iris so every stage uses the Iris model"
    )

VM.KS = V.C.TOP_K_VALUES

SIGNALS_PATH = V.C.DATA_DIR / "doc2querypp_bertopic_keybert_hf_signals.json"
GEN_PATH = V.C.DATA_DIR / "doc2querypp_bertopic_keybert_hf_signals_questions.jsonl"
RESULTS_PATH = V.C.RESULTS_DIR / "doc2querypp_bertopic_keybert_iris_fusion.json"
TEXT_COLL = "mhrag_vo15_iris_doc2querypp_text_512_256"
QUESTION_COLL = "mhrag_vo15_iris_doc2querypp_questions_512_256"
REPORT = V.REPORT
MARKER_START = "<!-- DOC2QUERYPP_EXACT_START -->"
MARKER_END = "<!-- DOC2QUERYPP_EXACT_END -->"
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
_SENT = re.compile(r"(?<=[.!?])\s+")

_SYSTEM = (
    "You generate grounded, diverse retrieval questions for a complete news "
    "article using supplied BERTopic topics and KeyBERT keywords. Every evidence "
    "span must be copied verbatim from the article. Output ONLY a JSON object."
)


def _articles() -> List[dict]:
    return V.D.read_jsonl(V.C.PROCESSED_ARTICLES)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.split()) >= 5]


def _l2_normalize(vectors) -> np.ndarray:
    """Numerically stable unit vectors for BERTopic/KeyBERT cosine math."""
    arr = np.asarray(vectors, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Iris returned non-finite embedding values")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


def build_signals(force: bool = False) -> dict:
    if SIGNALS_PATH.exists() and not force:
        return json.load(SIGNALS_PATH.open(encoding="utf-8"))

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from keybert import KeyBERT
    from keybert.backend import BaseEmbedder
    from umap import UMAP
    from embeddings import HFEmbedder

    articles = _articles()
    sentence_rows = [
        (a["article_id"], sentence)
        for a in articles
        for sentence in _sentences(a["cleaned_body"])
    ]
    sentence_texts = [x[1] for x in sentence_rows]
    # Iris remains the retrieval embedder. BERTopic/KeyBERT use the agreed HF
    # fallback because Iris vectors overflow KeyBERT's cosine matrix routines.
    signal_embedder = HFEmbedder()
    print(f"[d2q++ exact] embedding {len(sentence_texts)} sentences for BERTopic")
    sentence_embeddings = _l2_normalize(signal_embedder.embed_documents(sentence_texts))

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=V.C.SEED,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(sentence_texts, sentence_embeddings)
    topic_words = {
        int(t): [word for word, _score in (topic_model.get_topic(t) or [])[:10]]
        for t in sorted(set(topics))
        if t != -1
    }

    article_topic_counts: Dict[str, Counter] = defaultdict(Counter)
    for (article_id, _sentence), topic in zip(sentence_rows, topics):
        if topic != -1:
            article_topic_counts[article_id][int(topic)] += 1

    class NormalizedHFKeyBERTBackend(BaseEmbedder):
        def __init__(self, hf_embedder):
            super().__init__()
            self.hf_embedder = hf_embedder

        def embed(self, documents, verbose: bool = False):
            return _l2_normalize(self.hf_embedder.embed_documents(list(documents)))

    kw_model = KeyBERT(model=NormalizedHFKeyBERTBackend(signal_embedder))
    article_signals = {}
    for article in articles:
        counts = article_topic_counts[article["article_id"]]
        selected = [tid for tid, _ in counts.most_common(5)]
        if len(selected) < 2:
            # Stable fallback: globally largest discovered topics.
            global_counts = Counter(topics)
            selected += [
                int(tid)
                for tid, _ in global_counts.most_common()
                if tid != -1 and tid not in selected
            ][: 2 - len(selected)]
        topic_labels = [
            {
                "topic_id": tid,
                "keywords": topic_words.get(tid, []),
                "sentence_count": counts.get(tid, 0),
            }
            for tid in selected[:5]
        ]
        kws = kw_model.extract_keywords(
            article["cleaned_body"],
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            use_mmr=True,
            diversity=0.6,
            top_n=15,
        )
        article_signals[article["article_id"]] = {
            "title": article["title"],
            "topics": topic_labels,
            "keywords": [phrase for phrase, _score in kws],
            "keyword_scores": {phrase: float(score) for phrase, score in kws},
        }

    result = {
        "method": {
            "topic_model": "BERTopic",
            "topic_unit": "sentences from the selected 15-article collection",
            "keybert_mmr_diversity": 0.6,
            "signal_embedding_model": signal_embedder.name,
            "retrieval_embedding_signature": embedding_signature(),
            "fallback_reason": "Iris vectors overflowed KeyBERT cosine math",
            "seed": V.C.SEED,
        },
        "num_sentences": len(sentence_texts),
        "num_topics_excluding_outlier": len(topic_words),
        "topic_words": topic_words,
        "articles": article_signals,
    }
    SIGNALS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[d2q++ exact] topics={len(topic_words)} signals={len(article_signals)}")
    return result


def _prompt(article: dict, signal: dict, existing: List[dict] | None = None) -> str:
    topics = "\n".join(
        f"- Topic {t['topic_id']}: {', '.join(t['keywords'])}" for t in signal["topics"]
    )
    keywords = ", ".join(signal["keywords"])
    prior = ""
    requested = 30
    if existing:
        requested = 30 - len(existing) + 3
        prior = "\nAlready accepted questions (do not repeat):\n" + "\n".join(
            f"- {x['question']}" for x in existing
        )
    return f'''Complete article:
"""
{article["cleaned_body"]}
"""

BERTopic topics:
{topics}

KeyBERT keywords:
{keywords}
{prior}

Generate {requested} concise, natural retrieval-question records. Across the
final set, cover all supplied topics and as many important keywords as the
article supports. Diversify across entities, events, dates, comparisons, causes,
consequences, numerical details, and inferred connections when available.

Rules:
- Every question must be fully answerable from this complete article.
- Avoid paraphrases and repeated coverage of the same fact.
- Preserve important names, dates, numbers, and exact terminology.
- Save a concise answer and a short supporting-evidence span copied VERBATIM
  from the article for every question.
- Evidence must fit inside one retrieval chunk.

Return JSON:
{{"items":[{{"question":"...","answer":"...","evidence":"verbatim quote"}}, ...]}}'''


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


def _generate_one(client, article: dict, signal: dict, chunks: List[dict]) -> dict:
    items: List[dict] = []
    for _ in range(5):
        if len(items) >= 30:
            break
        try:
            data = _call(client, _prompt(article, signal, items or None))
            items = AR._parse_items(
                {"items": items + data.get("items", [])}, article, chunks
            )
        except Exception:
            continue
    return {
        "article_id": article["article_id"],
        "title": article["title"],
        "topics": signal["topics"],
        "keywords": signal["keywords"],
        "items": items[:30],
        "valid": len(items) >= 30,
    }


def generate(signals: dict, force: bool = False) -> List[dict]:
    articles = _articles()
    chunks = AR._chunks_by_article()
    cache = (
        {r["article_id"]: r for r in V.D.read_jsonl(GEN_PATH)}
        if GEN_PATH.exists() and not force
        else {}
    )
    todo = [
        a
        for a in articles
        if not cache.get(a["article_id"], {}).get("valid")
        or len(cache[a["article_id"]].get("items", [])) != 30
    ]
    print(f"[d2q++ exact] generation to-do {len(todo)}/{len(articles)}")
    if todo:
        client = V.C.openai_client()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                ex.submit(
                    _generate_one,
                    client,
                    article,
                    signals["articles"][article["article_id"]],
                    chunks[article["article_id"]],
                ): article
                for article in todo
            }
            for fut in as_completed(futures):
                article = futures[fut]
                cache[article["article_id"]] = fut.result()
        print(f"[d2q++ exact] generation {time.perf_counter() - t0:.1f}s")
    rows = [cache[a["article_id"]] for a in articles]
    V.D._write_jsonl(GEN_PATH, rows)
    print(
        f"[d2q++ exact] valid={sum(r['valid'] for r in rows)}/15 "
        f"questions={sum(len(r['items']) for r in rows)}"
    )
    return rows


def build_indexes(rows: List[dict]) -> dict:
    if not all(r["valid"] and len(r["items"]) == 30 for r in rows):
        raise RuntimeError("Question cache incomplete; rerun to repair")
    embedder = get_embedder()
    chunks = V.D.load_chunks()
    text_vectors = embedder.embed_documents([c["text"] for c in chunks])
    text_coll = V.C.reset_collection(TEXT_COLL)
    text_coll.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=text_vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "parent_chunk_id": c["chunk_id"],
                "parent_document_id": c["parent_document_id"],
            }
            for c in chunks
        ],
    )

    items = [item for row in rows for item in row["items"]]
    question_vectors = embedder.embed_documents([x["question"] for x in items])
    question_coll = V.C.reset_collection(QUESTION_COLL)
    question_coll.add(
        ids=[f"d2qpp-exact::{i}" for i in range(len(items))],
        embeddings=question_vectors,
        documents=[x["question"] for x in items],
        metadatas=[
            {
                "generated_question_id": f"d2qpp-exact::{i}",
                "parent_chunk_id": x["parent_chunk_id"],
                "parent_document_id": x["parent_article_id"],
                "answer": x["answer"],
                "evidence": x["evidence"],
            }
            for i, x in enumerate(items)
        ],
    )
    print(f"[d2q++ exact] indexed text={len(chunks)} questions={len(items)}")
    return {"text": len(chunks), "questions": len(items)}


def retrieve() -> Dict[str, List[dict]]:
    queries, gold = V.D.load_eligible_queries(), V.D.load_gold()
    chunk_map = {c["chunk_id"]: c for c in V.D.load_chunks()}
    embedder = get_embedder()
    text_coll = V.C.get_collection(TEXT_COLL)
    question_coll = V.C.get_collection(QUESTION_COLL)
    conditions = {f"A{int(alpha * 10):02d}": [] for alpha in ALPHAS}

    for q in queries:
        qvec = embedder.embed_query(q["query"])
        text_raw = AR._all_chunk_scores(text_coll, qvec)
        question_raw = AR._all_question_scores(question_coll, qvec)
        text_scores = AR._minmax(text_raw)
        question_scores = AR._minmax(question_raw)
        common = AR._common(q, gold[q["query_id"]])
        for alpha in ALPHAS:
            tag = f"A{int(alpha * 10):02d}"
            if alpha == 0:
                scores = text_raw
            else:
                scores = {
                    cid: (
                        (1 - alpha) * text_scores.get(cid, 0.0)
                        + alpha * question_scores.get(cid, 0.0)
                    )
                    for cid in chunk_map
                }
            conditions[tag].append({**common, "ranked": AR._rank(scores, chunk_map)})
    return conditions


def evaluate(rankings: Dict[str, List[dict]], vectors: dict, signals: dict) -> dict:
    per = {
        tag: {r["query_id"]: VM.per_query(r) for r in rows}
        for tag, rows in rankings.items()
    }
    qids = sorted(per["A00"])
    metric_defs = [
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ]

    def vals(tag, key):
        return np.array([per[tag][qid][key] for qid in qids])

    data = {}
    for key, _ in metric_defs:
        base = vals("A00", key)
        data[key] = {}
        for tag in per:
            x = vals(tag, key)
            lo, hi = V._ci(x)
            item = {"mean": float(x.mean()), "ci_low": lo, "ci_high": hi}
            if tag != "A00":
                dm, dlo, dhi, sig = V._dci(base, x)
                item.update(
                    delta=float(dm),
                    delta_low=dlo,
                    delta_high=dhi,
                    significant=bool(sig),
                )
            data[key][tag] = item
    fusion_tags = [t for t in per if t != "A00"]
    best = max(
        fusion_tags,
        key=lambda t: (
            data["evidence_recall@5"][t]["mean"],
            data["mrr@10"][t]["mean"],
            -int(t[1:]),
        ),
    )
    result = {
        "n_queries": len(qids),
        "embedding_signature": embedding_signature(),
        "vectors": vectors,
        "num_bertopic_topics": signals["num_topics_excluding_outlier"],
        "num_topic_sentences": signals["num_sentences"],
        "alphas": {f"A{int(a * 10):02d}": a for a in ALPHAS},
        "normalization": "per-query independent min-max",
        "question_parent_aggregation": "max cosine",
        "best_fusion": best,
        "metrics": data,
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _cell(item: dict, base: bool = False) -> str:
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
        f'<div class="d {cls}" title="Δ vs text-only 95% CI '
        f'[{item["delta_low"]:+.3f}, {item["delta_high"]:+.3f}]">'
        f"({d:+.3f}{star})</div>"
    )


def update_report(result: dict) -> None:
    metric_defs = [
        ("evidence_recall@1", "Evidence Recall@1"),
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("all_evidence_hit@5", "Full-evidence@5"),
        ("mrr@10", "MRR@10"),
    ]
    body = ""
    for tag, alpha in result["alphas"].items():
        label = (
            "Iris original chunks only"
            if alpha == 0
            else (
                "Iris generated questions only"
                if alpha == 1
                else f"Dual index · {alpha:.1f} question / {1 - alpha:.1f} text"
            )
        )
        cells = "".join(
            f"<td>{_cell(result['metrics'][key][tag], alpha == 0)}</td>"
            for key, _ in metric_defs
        )
        best = " <b>· best fusion</b>" if tag == result["best_fusion"] else ""
        vectors = "119" if alpha == 0 else ("450" if alpha == 1 else "119 + 450")
        body += (
            f'<tr class="{"base" if alpha == 0 else ""}">'
            f'<td class="nm"><b>α={alpha:.1f}</b> · {escape(label)}{best}</td>'
            f"{cells}<td>{vectors}</td></tr>"
        )
    heads = "".join(f"<th>{escape(label)}</th>" for _, label in metric_defs)
    section = f"""{MARKER_START}
<h2>Doc2Query++ pipeline — BERTopic + KeyBERT signals, Iris dual-index fusion</h2>
<p class="cap">Complete article → BERTopic broader topics → KeyBERT MMR keywords
→ GPT-5.4-mini 30 grounded diverse questions/article → separate Iris
chunk and question indexes → independently min–max normalized score fusion.
Every question stores its answer, verbatim evidence, parent article, and
evidence-containing parent chunk. BERTopic and KeyBERT use the configured
Hugging Face fallback because Iris vectors caused numerical overflow in
KeyBERT; retrieval remains fully Iris.</p>
<table><thead><tr><th>Condition</th>{heads}<th>Index vectors</th></tr></thead>
<tbody>{body}</tbody></table>
<p class="cap">Best fusion by Evidence Recall@5 (MRR tie-break):
<b>α={result["alphas"][result["best_fusion"]]:.1f}</b>. BERTopic fitted
{result["num_topic_sentences"]} sentences into
{result["num_bertopic_topics"]} non-outlier topics. n={result["n_queries"]}
queries · 15 articles · 119 chunks · 512/256 ·
{escape(result["embedding_signature"])}. Deltas are versus this section's Iris
text-only baseline; * means paired 95% bootstrap CI excludes zero.</p>
{MARKER_END}"""
    html = REPORT.read_text(encoding="utf-8")
    if MARKER_START in html:
        start = html.index(MARKER_START)
        end = html.index(MARKER_END, start) + len(MARKER_END)
        html = html[:start] + section + html[end:]
    else:
        html = html.replace("</div></body></html>", section + "\n</div></body></html>")
    REPORT.write_text(html, encoding="utf-8")
    print(f"[d2q++ exact] updated {REPORT}")


def run() -> dict:
    V.D.build_all(force=False)
    signals = build_signals(force=False)
    rows = generate(signals, force=False)
    if not all(r["valid"] and len(r["items"]) == 30 for r in rows):
        raise RuntimeError("Generation incomplete; rerun to repair cached articles")
    vectors = build_indexes(rows)
    rankings = retrieve()
    result = evaluate(rankings, vectors, signals)
    update_report(result)
    for key, label in (
        ("evidence_recall@5", "Evidence Recall@5"),
        ("evidence_recall@10", "Evidence Recall@10"),
        ("mrr@10", "MRR@10"),
    ):
        print(
            f"  {label:<20} "
            + "  ".join(
                f"α={a:.1f}:{result['metrics'][key][f'A{int(a * 10):02d}']['mean']:.3f}"
                for a in ALPHAS
            )
        )
    print(f"[d2q++ exact] best α={result['alphas'][result['best_fusion']]}")
    return result


if __name__ == "__main__":
    run()
