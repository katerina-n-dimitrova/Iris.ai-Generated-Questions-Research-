# Context-Enrichment RAG — Evaluation Harness

Evaluate how **context enrichment** affects a Retrieval-Augmented Generation
(RAG) pipeline across four document types:

| Input type            | Dataset             | Hugging Face id                |
|-----------------------|---------------------|--------------------------------|
| Structured sci. text  | SciFact             | `allenai/scifact`              |
| Unstructured biomed.  | BEIR NFCorpus       | `BeIR/nfcorpus`                |
| Tables                | WikiTableQuestions  | `stanfordnlp/wikitablequestions` |
| Charts / graphs       | ChartQA             | `HuggingFaceM4/ChartQA`        |

We compare two conditions per dataset:

- **Condition A — Baseline RAG:** embed raw content only.
- **Condition B — Context-Enriched RAG:** embed raw content + added context
  (titles, positions, headers, key entities, generated summaries, …).

…and measure: **(1)** retrieval quality, **(2)** answer quality,
**(3)** offline encoding latency, **(4)** online ChromaDB query latency,
**(5)** index size / storage cost.

> This is a first end-to-end version designed to run on a **small subset**
> (`MAX_DATASET_SAMPLES`, default 300). It is modular so you can extend it to
> scanned PDFs, slide decks, spreadsheets, and multimodal documents later.

---

## Embedding model

By default embeddings use the Hugging Face model
**`Octen/Octen-Embedding-0.6B`** via `sentence-transformers` (runs locally — no
embedding API cost). Switch backends in `.env`:

```env
EMBEDDING_BACKEND=huggingface          # default
HF_EMBEDDING_MODEL=Octen/Octen-Embedding-0.6B
# EMBEDDING_BACKEND=openai             # to use OPENAI_EMBEDDING_MODEL instead
```

The chat model (`OPENAI_CHAT_MODEL`) is still used for optional LLM enrichment,
answer generation, and the LLM-as-judge metric.

## ChromaDB — local (default) or cloud

By default this project uses **ChromaDB in local persistent mode**
(`chromadb.PersistentClient` → `./chroma_indexes/` on disk). In local mode there
is **no API key** to configure — leave the cloud fields as placeholders.

To run indexes/queries against **Chroma Cloud** instead, set `CHROMA_MODE=cloud`
and paste your credentials in `.env`:

```env
CHROMA_MODE=cloud
CHROMA_API_KEY=ck-...your_key...
CHROMA_TENANT=your_tenant_id
CHROMA_DATABASE=your_database_name
```

The same scripts then build collections and run retrieval/latency analysis
against the hosted database — all metrics are still computed locally from the
query results. Note: `index_size_mb` is reported as `-1` in cloud mode (on-disk
size isn't measurable remotely); all other latency/quality metrics work
identically. The active backend is printed at the top of each run
(`Chroma backend: cloud:<tenant>/<database>`).

---

## 1. Install dependencies

```bash
cd context-enrichment-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First run will also download NLTK data on demand if needed:

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 2. Create the `.env` file

```bash
cp .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY` (only needed for answer generation,
LLM enrichment, or the LLM judge — **not** required for retrieval + latency
experiments with the default HF embedder). `.env` is gitignored so your key is
never committed.

## 3. Download the datasets

Pulls a capped subset of each dataset into `data/raw/`:

```bash
python src/download_datasets.py                       # all four
python src/download_datasets.py --datasets scifact    # one
python src/download_datasets.py --max-samples 100
```

## 4. Preprocess baseline and enriched chunks

Each dataset produces `processed/<dataset>_baseline.jsonl`,
`processed/<dataset>_enriched.jsonl`, and `processed/<dataset>_queries.jsonl`.

```bash
# both conditions, per dataset:
python src/preprocess_scifact.py
python src/preprocess_nfcorpus.py
python src/preprocess_wikitablequestions.py
python src/preprocess_chartqa.py

# or orchestrate across all datasets by condition:
python src/create_baseline_chunks.py
python src/create_enriched_chunks.py            # cheap offline enrichment
python src/create_enriched_chunks.py --use-llm  # LLM-generated summaries
```

Each JSONL row:

```json
{
  "id": "scifact_123_s0_baseline",
  "dataset": "scifact",
  "input_type": "structured_text",
  "condition": "baseline",
  "text_for_embedding": "...",
  "original_text": "...",
  "metadata": {"source_id": "123", "question_id": null, "title": "...",
               "page": "1/8", "row_id": null, "chart_id": null}
}
```

## 5. Build ChromaDB indexes

Creates one collection per dataset×condition (`scifact_baseline`,
`scifact_enriched`, …) and logs **offline encoding latency + index size**.

```bash
python src/build_chroma_indexes.py
python src/build_chroma_indexes.py --datasets scifact --conditions baseline
```

→ `results/latency_logs/offline_indexing.csv`

## 6. Run retrieval experiments

Embeds each query, searches Chroma, assembles context, logs **online query
latency**, and saves retrieved chunks for evaluation.

```bash
python src/run_retrieval_experiments.py
python src/run_retrieval_experiments.py --top-k 5
```

→ `results/latency_logs/online_query_latency.csv`
→ `results/retrieval_metrics/retrieved_<dataset>_<condition>.jsonl`

### (optional) Generate + evaluate answers

```bash
python src/run_answer_generation.py            # needs OPENAI_API_KEY
python src/run_answer_generation.py --dry-run  # pipeline test, no API calls
python src/evaluate_answers.py                 # exact match + token F1
python src/evaluate_answers.py --use-llm-judge # + LLM 0/1 correctness
```

→ `results/answer_metrics/answers_*.jsonl`, `answer_scores.csv`

## 7. Measure offline encoding latency

Logged automatically in step 5. Columns: `num_documents`, `num_chunks`,
`avg_tokens_per_chunk`, `total_embedding_time_seconds`,
`avg_embedding_time_per_chunk_ms`, `chroma_add_time_seconds`,
`total_indexing_time_seconds`, `index_size_mb`.

## 8. Measure online query latency

Logged automatically in step 6. Columns: `query_embedding_latency_ms`,
`chroma_search_latency_ms`, `context_assembly_latency_ms`,
`total_retrieval_latency_ms`, `retrieved_chunk_ids`.

## 9. View the final results

Compute retrieval metrics and a consolidated latency/storage summary:

```bash
python src/evaluate_retrieval.py     # Recall@5/10, Precision@5, MRR, nDCG@10, Hit@5
python src/measure_latency.py        # baseline vs enriched: encode/query latency + size
```

Outputs:
- `results/retrieval_metrics/retrieval_scores.csv`
- `results/answer_metrics/answer_scores.csv`
- `results/latency_logs/latency_summary.csv`

---

## One-shot quickstart

```bash
python src/download_datasets.py --max-samples 100
python src/create_baseline_chunks.py --max-samples 100
python src/create_enriched_chunks.py --max-samples 100
python src/build_chroma_indexes.py
python src/run_retrieval_experiments.py
python src/evaluate_retrieval.py
python src/measure_latency.py
```

## Project layout

```text
context-enrichment-rag/
  data/{raw,processed}/          # downloaded + chunked data
  chroma_indexes/                # persistent Chroma collections
  results/{latency_logs,retrieval_metrics,answer_metrics}/
  src/
    config.py                    # paths, dataset registry, env knobs
    embeddings.py                # HF (Octen) / OpenAI embedding backends
    common.py                    # JSONL I/O, record schema, enrichment helpers
    download_datasets.py
    preprocess_*.py              # one per dataset (baseline + enriched + queries)
    create_baseline_chunks.py    # orchestrators
    create_enriched_chunks.py
    build_chroma_indexes.py      # + offline latency log
    run_retrieval_experiments.py # + online latency log
    run_answer_generation.py
    evaluate_retrieval.py
    evaluate_answers.py
    measure_latency.py
```

## Extending to new document types

Add a `preprocess_<type>.py` exposing `build_documents(use_llm, max_samples)`
(returns `{"baseline": [...], "enriched": [...]}`) and
`build_queries(max_samples)`, register the dataset in `config.DATASETS`, and add
it to the `MODULES` maps in the two orchestrators. For scanned PDFs / slides /
charts, implement the text/feature extraction inside that module (e.g. the
`extract_chart_text()` hook in `preprocess_chartqa.py`) — the indexing,
retrieval, latency, and evaluation stages need no changes.

## Notes / limitations

- **ChartQA** ships images without bundled OCR/caption text. This first version
  uses the answer label as a textual stand-in and leaves placeholder fields
  (chart type, axes, legend) in the enriched format. Plug a vision/OCR model
  into `extract_chart_text()` for real chart signal.
- Relevance for retrieval metrics is judged at the **source-document** level
  (`metadata.source_id` ∈ `gold_source_ids`).
- Answer metrics start simple (exact match, token F1, optional LLM judge);
  `faithfulness`, `citation_accuracy`, `answer_relevance` are stubbed with a
  ragas hook in `evaluate_answers.compute_ragas_metrics()`.
```
