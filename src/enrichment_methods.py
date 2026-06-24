"""
Context-enrichment methods to compare per document type.

For each dataset we test 5 conditions:
    baseline, method 1, method 2, method 3, combined_best

Design
------
Each dataset has a `units_<dataset>()` builder that parses the raw data once
into a list of "units" (dicts holding every field any method might need) plus
the eval queries. Each enrichment method is a small function
`enrich_<dataset>_<method>(unit) -> text_for_embedding`. A registry
(DATASET_METHODS) maps condition names to those functions so the runner can
iterate generically. This keeps the design modular and easy to extend to new
document types (scanned PDFs, slides, spreadsheets, multimodal…).

LLM-backed methods reuse common.llm_summary (falls back to cheap heuristics if
no API). Fields that genuinely require OCR/vision/derendering (ChartQA
chart-to-table, axis metadata) or that the source doesn't provide (formula
surrounding text) are emitted as documented placeholders — we never hallucinate
values.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Tuple

import common
import config

# reuse existing dataset parsing
import preprocess_scifact as ps
import preprocess_nfcorpus as pn
import preprocess_wikitablequestions as pw
import preprocess_chartqa as pc
import preprocess_formulareasoning as pf


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p and p.strip())


def _parallel(fn, items, workers: int = 8):
    """Run fn over items concurrently (LLM enrichment is latency-bound)."""
    from concurrent.futures import ThreadPoolExecutor
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


# =========================================================================== #
# SciFact — structured scientific text
# =========================================================================== #
def units_scifact(n_queries: int, n_distractors: int, use_llm: bool):
    corpus = ps._corpus_index()
    qtext, gold = common.load_beir_queries_gold(ps.SPEC.raw_dir)
    selected, index_ids = common.select_eval_subset(
        qtext, gold, corpus.keys(), n_queries, n_distractors)
    client = config.get_openai_client() if use_llm else None
    # one doc-context per document, computed concurrently and reused per sentence
    docs = []
    for doc_id in sorted(index_ids):
        row = corpus[doc_id]
        title = str(row.get("title") or "").strip()
        sents = ps._abstract_sentences(row)
        docs.append((doc_id, title, sents, " ".join(sents)))
    if use_llm:
        ctxs = _parallel(lambda d: common.llm_summary(
            client, f"Title: {d[1]}\nAbstract: {d[3]}",
            kind="scientific paper (say what it is about in one sentence)"), docs)
    else:
        ctxs = [f"a scientific abstract titled '{d[1]}'" if d[1]
                else "a scientific abstract" for d in docs]

    units = []
    for (doc_id, title, sents, abstract_full), llm_ctx in zip(docs, ctxs):
        n = len(sents)
        for i, s in enumerate(sents):
            units.append({
                "source_id": doc_id, "title": title, "text": s,
                "abstract_full": abstract_full,
                "prev": sents[i - 1] if i > 0 else "",
                "next": sents[i + 1] if i < n - 1 else "",
                "pos": f"{i+1}/{n}", "llm_ctx": llm_ctx,
                "chunk_key": f"scifact_{doc_id}_s{i}",
            })
    queries = [{"query_id": q, "dataset": "scifact", "text": t,
                "gold_source_ids": g} for q, t, g in selected]
    return units, queries


def enrich_scifact_baseline(u): return u["text"]

def enrich_scifact_title_abstract_context(u):
    return _join(f"Paper title: {u['title']}",
                 f"Abstract context: {u['abstract_full'][:600]}",
                 f"Original chunk: {u['text']}")

def enrich_scifact_neighboring_context(u):
    return _join(f"Previous sentence: {u['prev']}" if u['prev'] else "",
                 f"Current chunk: {u['text']}",
                 f"Next sentence: {u['next']}" if u['next'] else "")

def enrich_scifact_llm_context(u):
    return _join(f"Document-aware context: This chunk comes from {u['llm_ctx']}",
                 f"Original chunk: {u['text']}")

def enrich_scifact_combined(u):
    return _join(f"Paper title: {u['title']}",
                 f"Abstract context: {u['abstract_full'][:400]}",
                 f"Document-aware context: This chunk comes from {u['llm_ctx']}",
                 f"Previous sentence: {u['prev']}" if u['prev'] else "",
                 f"Next sentence: {u['next']}" if u['next'] else "",
                 f"Original chunk: {u['text']}")


# =========================================================================== #
# NFCorpus — unstructured biomedical text
# =========================================================================== #
def units_nfcorpus(n_queries: int, n_distractors: int, use_llm: bool):
    corpus = pn._corpus_index()
    qtext, gold = common.load_beir_queries_gold(pn.SPEC.raw_dir)
    selected, index_ids = common.select_eval_subset(
        qtext, gold, corpus.keys(), n_queries, n_distractors)
    client = config.get_openai_client() if use_llm else None
    # collect all chunks first, then run the two LLM enrichments concurrently
    chunks = []
    for doc_id in sorted(index_ids):
        row = corpus[doc_id]
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        for ci, chunk in enumerate(common.chunk_text(text)):
            chunks.append((doc_id, title, chunk, ci))
    if use_llm:
        gqs = _parallel(lambda c: _gen_questions(client, c[2]), chunks)
        sums = _parallel(lambda c: common.llm_summary(
            client, c[2], kind="biomedical passage in plain language"), chunks)
    else:
        gqs = ["- (enable --use-llm to generate questions)"] * len(chunks)
        sums = [common.cheap_summary(c[2]) for c in chunks]

    units = []
    for (doc_id, title, chunk, ci), gen_q, summ in zip(chunks, gqs, sums):
        units.append({
            "source_id": doc_id, "title": title, "text": chunk,
            "keywords": common.cheap_keywords(f"{title}. {chunk}"),
            "gen_questions": gen_q, "plain_summary": summ,
            "chunk_key": f"nfcorpus_{doc_id}_c{ci}",
        })
    queries = [{"query_id": q, "dataset": "nfcorpus", "text": t,
                "gold_source_ids": g} for q, t, g in selected]
    return units, queries


def _gen_questions(client, text: str, n: int = 3) -> str:
    if client is None:
        return "- (enable --use-llm to generate questions)"
    try:
        r = client.chat.completions.create(
            model=config.OPENAI_CHAT_MODEL, temperature=0.0, max_tokens=120,
            messages=[{"role": "system", "content":
                       "List the questions a reader could answer from the passage. "
                       "Output 3 short questions as '- ' bullets, nothing else."},
                      {"role": "user", "content": text[:2000]}])
        return r.choices[0].message.content.strip()
    except Exception:
        return "- (question generation failed)"


def enrich_nfcorpus_baseline(u): return u["text"]

def enrich_nfcorpus_generated_questions(u):
    return _join("Likely questions this passage answers:", u["gen_questions"],
                 f"Original passage: {u['text']}")

def enrich_nfcorpus_keywords_entities(u):
    return _join(f"Biomedical keywords/entities: {u['keywords']}",
                 f"Original passage: {u['text']}")

def enrich_nfcorpus_plain_summary(u):
    return _join(f"Plain-language summary: {u['plain_summary']}",
                 f"Original passage: {u['text']}")

def enrich_nfcorpus_combined(u):
    return _join("Likely questions this passage answers:", u["gen_questions"],
                 f"Biomedical keywords/entities: {u['keywords']}",
                 f"Plain-language summary: {u['plain_summary']}",
                 f"Original passage: {u['text']}")


# =========================================================================== #
# WikiTableQuestions — tables
# =========================================================================== #
def units_wikitablequestions(n_queries: int, n_distractors: int, use_llm: bool):
    from common import read_jsonl
    test_path = pw.SPEC.raw_dir / "test.jsonl"
    rows = list(read_jsonl(test_path))[:max(n_queries, n_distractors)]
    units, queries, seen = [], [], set()
    for idx, row in enumerate(rows):
        table = row.get("table") or {}
        tid = pw._table_id(table, fallback=f"table_{idx}")
        title = str(table.get("name") or row.get("page_title") or tid).strip()
        headers = [str(h) for h in (table.get("header") or [])]
        if tid not in seen:
            seen.add(tid)
            for ri, data_row in enumerate(table.get("rows") or []):
                linear = pw._linearize_row(headers, data_row)
                if not linear:
                    continue
                units.append({
                    "source_id": tid, "title": title,
                    "headers": ", ".join(headers), "text": linear,
                    "row_summary": "In table '%s', " % title + "; ".join(
                        f"{h} is {v}" for h, v in zip(headers, data_row) if str(v).strip()),
                    "chunk_key": f"wtq_{tid}_r{ri}".replace("/", "_"),
                })
        q = str(row.get("question") or "").strip()
        if q and len(queries) < n_queries:
            ans = row.get("answers")
            queries.append({"query_id": str(row.get("id") or f"wtq_q{idx}"),
                            "dataset": "wikitablequestions", "text": q,
                            "gold_source_ids": [tid],
                            "gold_answer": ", ".join(map(str, ans)) if isinstance(ans, list) else str(ans or "")})
    return units, queries


def enrich_wikitable_baseline(u): return u["text"]
def enrich_wikitable_headers(u):
    return _join(f"Columns: {u['headers']}", f"Row: {u['text']}")
def enrich_wikitable_title(u):
    return _join(f"Page/table title: {u['title']}", f"Original row: {u['text']}")
def enrich_wikitable_row_summary(u):
    return _join(f"Row summary: {u['row_summary']}", f"Original row: {u['text']}")
def enrich_wikitable_combined(u):
    return _join(f"Page/table title: {u['title']}", f"Columns: {u['headers']}",
                 f"Row summary: {u['row_summary']}", f"Original row: {u['text']}")


# =========================================================================== #
# ChartQA — charts / graphs (limited textual signal; placeholders documented)
# =========================================================================== #
CHART_TABLE_PLACEHOLDER = ("[chart-to-table extraction unavailable: requires "
                           "OCR/vision/chart-derendering on the chart image]")
CHART_AXIS_PLACEHOLDER = ("[axis/legend/title metadata unavailable: requires "
                          "vision parsing of the chart image]")


def units_chartqa(n_queries: int, n_distractors: int, use_llm: bool):
    from common import read_jsonl
    test_path = pc.SPEC.raw_dir / "test.jsonl"
    rows = list(read_jsonl(test_path))[:max(n_queries, n_distractors)]
    client = config.get_openai_client() if use_llm else None
    charts, queries = [], []
    for idx, row in enumerate(rows):
        cid = f"chartqa_{idx}"
        fields = pc.extract_chart_text(row)
        ocr = (fields.get("ocr_text") or "").strip() or "(no extractable chart text)"
        charts.append((cid, fields.get("title"), ocr))
        if len(queries) < n_queries:
            q = str(row.get("query") or "").strip()
            if q:
                queries.append({"query_id": f"chartqa_q{idx}",
                                "dataset": "chartqa", "text": q,
                                "gold_source_ids": [cid],
                                "gold_answer": str(row.get("label") or "")})
    if use_llm:
        summaries = _parallel(lambda c: common.llm_summary(client, c[2], kind="chart"), charts)
    else:
        summaries = [common.cheap_summary(c[2]) for c in charts]
    units = [{"source_id": cid, "title": title, "text": ocr,
              "chart_table": CHART_TABLE_PLACEHOLDER,
              "axis_meta": CHART_AXIS_PLACEHOLDER, "summary": summ,
              "chunk_key": cid}
             for (cid, title, ocr), summ in zip(charts, summaries)]
    return units, queries


def enrich_chartqa_baseline(u): return u["text"]
def enrich_chartqa_data_table(u):
    return _join("Extracted chart data:", u["chart_table"],
                 f"Original chart text: {u['text']}")
def enrich_chartqa_axis_metadata(u):
    return _join(u["axis_meta"], f"Original chart text: {u['text']}")
def enrich_chartqa_summary(u):
    return _join(f"Chart summary: {u['summary']}",
                 f"Original chart text: {u['text']}")
def enrich_chartqa_combined(u):
    return _join("Extracted chart data:", u["chart_table"], u["axis_meta"],
                 f"Chart summary: {u['summary']}",
                 f"Original chart text: {u['text']}")


# =========================================================================== #
# FormulaReasoning — mathematical formulas
# =========================================================================== #
FORMULA_SURROUNDING_PLACEHOLDER = ("[surrounding text unavailable: the formula "
                                   "database stores standalone formulas]")


def units_formulareasoning(n_queries: int, n_distractors: int, use_llm: bool):
    formulas = pf._load("formulas.json")
    units = []
    for f in formulas:
        formula_en = str(f.get("formula", {}).get("en", "")).strip()
        if not formula_en:
            continue
        variables = pf._formula_vars(f)
        var_defs = "\n".join(f"{sym or nm} = {nm}" for nm, sym in variables)
        units.append({
            "source_id": str(f["key"]), "title": formula_en, "text": formula_en,
            "var_defs": var_defs,
            "var_list": ", ".join(nm for nm, _ in variables),
            "latex": _to_latex(formula_en),
            "structure": _formula_structure(formula_en),
            "surrounding": FORMULA_SURROUNDING_PLACEHOLDER,
            "chunk_key": f"formula_{f['key']}",
        })
    queries = pf.build_queries(n_queries)
    return units, queries


def _to_latex(formula: str) -> str:
    # bracket form e.g. "[Range]=[v]*[t]" -> readable pseudo-LaTeX
    s = formula.replace("*", r" \times ").replace("/", r" \div ")
    return re.sub(r"\[([^\]]+)\]", r"\\text{\1}", s)

def _formula_structure(formula: str) -> str:
    ops = sorted({c for c in formula if c in "+-*/=^()"})
    vars_ = re.findall(r"\[([^\]]+)\]", formula)
    return f"operators/symbols: {' '.join(ops)} | terms: {', '.join(vars_)}"


def enrich_formula_baseline(u): return u["text"]
def enrich_formula_surrounding_text(u):
    return _join(f"Context: {u['surrounding']}", f"Formula: {u['text']}")
def enrich_formula_variable_definitions(u):
    return _join(f"Formula: {u['text']}", "Variables:", u["var_defs"],
                 f"Quantities: {u['var_list']}")
def enrich_formula_structure(u):
    return _join(f"Formula in LaTeX: {u['latex']}",
                 f"Structure: {u['structure']}", f"Original: {u['text']}")
def enrich_formula_combined(u):
    return _join(f"Formula: {u['text']}", f"Context: {u['surrounding']}",
                 "Variables:", u["var_defs"],
                 f"Formula in LaTeX: {u['latex']}", f"Structure: {u['structure']}")


# =========================================================================== #
# Registry
# =========================================================================== #
# dataset -> { condition_name: enrich_fn }   (order: baseline, m1, m2, m3, combined)
DATASET_METHODS: Dict[str, Dict[str, Callable]] = {
    "scifact": {
        "baseline": enrich_scifact_baseline,
        "title_abstract_context": enrich_scifact_title_abstract_context,
        "neighboring_context": enrich_scifact_neighboring_context,
        "llm_generated_chunk_context": enrich_scifact_llm_context,
        "combined_best": enrich_scifact_combined,
    },
    "nfcorpus": {
        "baseline": enrich_nfcorpus_baseline,
        "generated_questions": enrich_nfcorpus_generated_questions,
        "keywords_entities": enrich_nfcorpus_keywords_entities,
        "plain_summary": enrich_nfcorpus_plain_summary,
        "combined_best": enrich_nfcorpus_combined,
    },
    "wikitablequestions": {
        "baseline": enrich_wikitable_baseline,
        "column_headers_per_row": enrich_wikitable_headers,
        "table_page_title": enrich_wikitable_title,
        "natural_language_row_summary": enrich_wikitable_row_summary,
        "combined_best": enrich_wikitable_combined,
    },
    "chartqa": {
        "baseline": enrich_chartqa_baseline,
        "chart_to_table_data": enrich_chartqa_data_table,
        "axis_legend_title_metadata": enrich_chartqa_axis_metadata,
        "chart_summary": enrich_chartqa_summary,
        "combined_best": enrich_chartqa_combined,
    },
    "formulareasoning": {
        "baseline": enrich_formula_baseline,
        "surrounding_text": enrich_formula_surrounding_text,
        "variable_definitions": enrich_formula_variable_definitions,
        "latex_structure": enrich_formula_structure,
        "combined_best": enrich_formula_combined,
    },
}

UNIT_BUILDERS: Dict[str, Callable] = {
    "scifact": units_scifact,
    "nfcorpus": units_nfcorpus,
    "wikitablequestions": units_wikitablequestions,
    "chartqa": units_chartqa,
    "formulareasoning": units_formulareasoning,
}

# which conditions require the LLM (so the runner can warn / cost)
LLM_CONDITIONS = {
    ("scifact", "llm_generated_chunk_context"), ("scifact", "combined_best"),
    ("nfcorpus", "generated_questions"), ("nfcorpus", "plain_summary"),
    ("nfcorpus", "combined_best"),
    ("chartqa", "chart_summary"), ("chartqa", "combined_best"),
}


def build_dataset(dataset: str, n_queries: int, n_distractors: int, use_llm: bool
                  ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Return ({condition: [records]}, queries) for one dataset."""
    units, queries = UNIT_BUILDERS[dataset](n_queries, n_distractors, use_llm)
    spec = config.DATASETS[dataset]
    conditions: Dict[str, List[Dict[str, Any]]] = {}
    for cond, fn in DATASET_METHODS[dataset].items():
        recs = []
        for u in units:
            text = fn(u)
            if not text or not text.strip():
                continue
            recs.append(common.make_record(
                chunk_id=f"{u['chunk_key']}__{cond}",
                dataset=dataset, input_type=spec.input_type, condition=cond,
                text_for_embedding=text, original_text=u["text"],
                source_id=u["source_id"], title=u.get("title")))
        conditions[cond] = recs
    return conditions, queries
