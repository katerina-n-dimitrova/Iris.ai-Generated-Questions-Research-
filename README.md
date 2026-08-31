# Context Enrichment for RAG — Adaptive Generated Questions

Does indexing **LLM-generated questions** alongside (or instead of) raw chunk
text improve RAG retrieval? This repo holds the final study: five controlled
retrieval conditions on two corpora, plus a combined-corpus robustness check
and a cross-model generator replication.

## The five conditions

Every condition shares the same chunks (1024 tokens / 128 overlap), the same
Iris dim-384 embeddings, the same BM25, and RRF k=60. Only the enrichment
changes.

| # | Condition | Enrichment | Dense retrieval |
|---|-----------|------------|-----------------|
| 1 | Baseline | none | chunk vectors |
| 2 | Adaptive generated questions 5–20 | `clamp(5, 20, round(facts × 0.5))` questions per chunk, from deduplicated atomic facts | 0.5/0.5 chunk / question fusion |
| 3 | Adaptive generated questions unbounded | `round(facts × 0.5)` per chunk, no bounds | 0.5/0.5 chunk / question fusion |
| 4 | Adaptive chunk + whole-article questions 5–20 | condition 2 + 5–20 questions per whole article | equal 1/3 chunk / chunk-question / article-question fusion |
| 5 | Adaptive chunk + whole-article questions unbounded | condition 3 + unbounded per whole article | equal 1/3 fusion |

The dense ranking is always fused with the chunk BM25 ranking by RRF (k=60).
Generator: `gpt-5.4-mini` @ temp 0.3 (except the baseline, which uses no LLM).

## The two datasets

| Dataset | Corpus | Chunks | Eval queries | Baseline MRR@10 | Best MRR@10 |
|---------|--------|--------|--------------|-----------------|-------------|
| MultiHop-RAG (`yixuantt/MultiHopRAG`) | 609 English news articles | 1,830 | 2,255 | 0.644 | 0.672 (cond. 4) |
| Yettel Bulgaria (self-built) | 340 Bulgarian telco documents | 963 | 2,255 (+301 null excluded) | 0.392 | 0.548 (cond. 5) |

## Layout

```
src/ragkit/          shared infrastructure: config, embedders (Iris/OpenAI/HF),
                     BM25 tokenizers, cosine helpers, RRF fusion, metrics, JSONL I/O
exp/multihoprag/     experiment 1 — the five conditions on MultiHop-RAG
exp/yettel_bg/       experiment 2 — the same five conditions on Yettel Bulgaria,
                     plus corpus/ (crawler, question generator, validator)
exp/combined/        experiment 3 — union-corpus interference study (949 docs)
exp/qwen_generator/  experiment 4 — conditions 1/2/4 re-run with Iris-hosted Qwen3.5-4B
data/raw/            source corpora (gitignored)
data/processed/      chunking + LLM-generation caches (gitignored; expensive to rebuild)
results/             metrics.json + rankings per experiment (vector caches gitignored)
report/              the deliverables: one self-contained HTML per experiment
```

## Setup

```bash
./prep.sh                      # uv venv + uv sync
cp .env.example .env           # then fill in OPENAI_API_KEY (and IRIS_* if needed)
```

All presentation runs used `EMBEDDING_BACKEND=iris` and
`OPENAI_CHAT_MODEL=gpt-5.4-mini`.

## Running

```bash
python exp/multihoprag/run.py               # all five MultiHop-RAG conditions, in order
python exp/yettel_bg/run_experiments.py     # all five Yettel conditions in one run
python exp/combined/run_combined_experiments.py
python exp/combined/combined_report.py      # re-render the rich combined report
```

Every stage is cached and resumable: re-running never regenerates questions or
re-embeds unless a cache is missing. With the shipped caches under
`data/processed/` and `results/`, full re-evaluation is offline and only
recomputes retrieval + metrics. Treat the generation caches
(`adaptive_generations.jsonl`, `article_question_generations*.jsonl`) as costly
artifacts — losing them means re-paying for all LLM calls.

There is no test suite — the smoke run is the test. Each experiment README
documents its own run order and outputs.

## Reports

| Report | Contents |
|--------|----------|
| `report/mhrag_full_no_question_baseline.html` | MultiHop-RAG condition 1 |
| `report/mhrag_adaptive_questions_full.html` | MultiHop-RAG, all five conditions |
| `report/yettel_bg_adaptive_questions.html` | Yettel, all five conditions |
| `report/combined_multihop_yettel_adaptive_questions.html` | union corpus, conditions 1–4 |
| `report/combined_multihop_yettel_four_experiments.html` | union corpus with delta / interference / leakage analysis |
| `report/mhrag_iris_qwen_5_20_full.html` | Qwen-generator replication (conditions 1, 2, 4) |
