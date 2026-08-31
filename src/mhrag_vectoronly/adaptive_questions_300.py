"""Run the two adaptive-question conditions on 300 MultiHop-RAG articles."""

from pathlib import Path

import adaptive_questions_100 as A


A.ARTICLE_COUNT = 300
A.MAX_WORKERS = 12
A.MAX_RETRIES = 8
A.DATA = A.ROOT / "data" / "processed" / "mhrag_adaptive_questions_300"
A.RESULTS = A.ROOT / "results" / "mhrag_adaptive_questions_300"
A.REPORT = A.ROOT / "report" / "mhrag_adaptive_questions_300.html"
for directory in (A.DATA, A.RESULTS, A.REPORT.parent):
    directory.mkdir(parents=True, exist_ok=True)

A.CHUNKS_PATH = A.DATA / "chunks.jsonl"
A.QUERIES_PATH = A.DATA / "queries.jsonl"
A.SUMMARY_PATH = A.DATA / "summary.json"
A.GEN_PATH = A.DATA / "adaptive_generations.jsonl"
A.CHUNK_VECTORS = A.RESULTS / "chunk_vectors_iris.json"
A.QUERY_VECTORS = A.RESULTS / "query_vectors_iris.json"
A.BOUNDED_VECTORS = A.RESULTS / "bounded_question_vectors_iris.json"
A.UNBOUNDED_VECTORS = A.RESULTS / "unbounded_question_vectors_iris.json"
A.METRICS = A.RESULTS / "metrics.json"
A.RANKINGS = A.RESULTS / "rankings.json"


if __name__ == "__main__":
    A.run()
