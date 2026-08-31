# MultiHop-RAG — dense-VECTOR-ONLY 15-article pilot

Does indexing **10 synthetic questions per chunk** improve retrieval of gold
evidence over indexing the **original chunks**? A closed-collection, 15-article
MultiHop-RAG pilot comparing two indexed vector representations of the *same*
chunks. **Vector search only** — cosine similarity in a local ChromaDB. No BM25,
sparse, keyword, hybrid, or RRF anywhere.

* **Condition A** (`baseline`): one embedding per original chunk (154 vectors).
* **Condition B** (`generated`): 10 generated-question embeddings per chunk
  (1540 vectors), mapped back to the parent chunk via max cosine similarity.

Everything else is held fixed: 15 articles, cleaning, token-chunk boundaries
(256/50/80), eligible queries, embedding model (Octen-Embedding-0.6B, cosine),
k=[1,3,4,5,10], gold mapping, answer LLM + prompt, seed 42.

## Run

```bash
cd src/mhrag_vectoronly
python run_vo.py                       # whole pipeline (cached/resumable)
python run_vo.py --stage inspect       # dataset + config + credentials sanity
python run_vo.py --stage prepare       # select 15 + clean + token-chunk + align gold
python run_vo.py --stage generate_questions
python run_vo.py --stage index         # build the two Chroma collections
python run_vo.py --stage retrieve
python run_vo.py --stage evaluate_retrieval
python run_vo.py --stage evaluate_generation
python run_vo.py --stage analyze_failures
python run_vo.py --stage report        # -> report/multihoprag_vectoronly_results.html
```

Credentials come only from the project `.env` (`OPENAI_API_KEY`,
`EMBEDDING_BACKEND`/`HF_EMBEDDING_MODEL`). Config knobs are in
`config/mhrag_vectoronly.yaml`. A dedicated **local** Chroma `PersistentClient`
is used regardless of `CHROMA_MODE`, so no cloud keys are needed.

## Modules

| file | stage |
|------|-------|
| `vo_config.py` | paths, knobs (YAML), local Chroma client |
| `vo_data.py` | query-first 15-article selection, cleaning, token chunking, gold alignment |
| `vo_generate.py` | 10 questions/chunk (structured JSON) + validation, cached |
| `vo_index.py` | two local Chroma collections (chunk vectors, question vectors) |
| `vo_retrieval.py` | dense-only retrieval for A and B (query vectors never stored) |
| `vo_metrics.py` | Hit/EvidenceRecall/Precision/MRR/MAP/nDCG/AllEvidenceHit/DocRecall + paired + bootstrap |
| `vo_answers.py` | grounded answer generation, EM/F1, evidence buckets |
| `vo_failure.py` | failure buckets + generated-question diagnostics + examples |
| `vo_report.py` | self-contained HTML report |
| `run_vo.py` | stage runner |

## Result (this pilot, n=84 eligible queries)

Generated-question indexing does **not** beat chunk indexing under dense-only
retrieval. Tied on Evidence Recall@5 (A=0.558 vs B=0.554, p=0.87); baseline is
slightly ahead at k=3–5 (Hit@4/@5 significant, p≈0.03) and finds the first
relevant chunk earlier. Answer EM 0.786 (A) vs 0.738 (B). Given ~10× more
vectors and extra offline generation cost, not justified on this pilot. This
reproduces the project's prior "questions-only loses early precision" finding.

> **Not** the official full-corpus MultiHop-RAG benchmark — a selected 15-article
> pilot. n is small; read significance, not point estimates.

## Generated-question similarity fallback

The 512/128 hierarchical dataset also has a cascade experiment that first
searches the 150 verified generated document questions. When the best question
cosine is below a configurable threshold, it falls back to unrestricted
Flat-hybrid retrieval over the original chunks:

```bash
python src/mhrag_vectoronly/vo_question_fallback_experiment.py --threshold 0.50
```

It reuses the persisted hierarchical indexes, caches query vectors, and writes
machine-readable rankings/metrics under
`results/mhrag_vectoronly/question_similarity_fallback/` plus the report
`report/mhrag_question_similarity_fallback.html`.
