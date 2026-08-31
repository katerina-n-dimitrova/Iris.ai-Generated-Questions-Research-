# MultiHop-RAG isolated question-type ablation

Seven independent dense-vector question indexes on the frozen 10-article,
96-chunk collection. Each type requests six grounded questions per chunk. Types
are never pooled. The unchanged original-chunk index is the baseline.

```bash
python src/mhrag_question_types/run_qt.py
python src/mhrag_question_types/run_qt.py --stage generate --force
```

Credentials come from environment variables via the repository configuration.
Outputs are written to `results/mhrag_question_types_10/` and the self-contained
report to `report/mhrag_question_types_10_results.html`.
