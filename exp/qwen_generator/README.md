# Qwen-generator replication (Iris-hosted)

Re-runs the enrichment conditions with the Iris-hosted `Qwen/Qwen3.5-4B`
chat endpoint (`ragkit/iris_llm_client.py`) instead of `gpt-5.4-mini`, to test
whether the enrichment effect survives a much smaller generator. The baseline
needs no LLM and is reused unchanged.

## MultiHop-RAG (complete)

```bash
python exp/qwen_generator/run_multihoprag.py [--batch 32]
```

Conditions 1, 2, and 4 only (no unbounded arms). Verified MRR@10:
0.644 (baseline) → 0.662 (5–20) → 0.654 (chunk + article 5–20) — the
enrichment still helps, but the article-question arm no longer adds on top.
Isolated caches: `data/processed/mhrag_iris_qwen_5_20_full/`,
`results/mhrag_iris_qwen_5_20_full/`, `report/mhrag_iris_qwen_5_20_full.html`.

## Yettel (in progress)

```bash
python exp/qwen_generator/run_yettel.py [--batch 24]
```

Same three conditions on the Yettel corpus with Bulgarian prompts. Only the
baseline rankings and chunk/query vectors exist so far
(`results/yettel_bg_experiments_iris_qwen/`); the generation stages have not
been completed yet.
