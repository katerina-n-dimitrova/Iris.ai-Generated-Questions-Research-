"""No-question baseline on the frozen 300-article adaptive subset."""

import baseline_100 as B


A = B.A
A.DATA = A.ROOT / "data" / "processed" / "mhrag_adaptive_questions_300"
A.RESULTS = A.ROOT / "results" / "mhrag_adaptive_questions_300"
A.REPORT = A.ROOT / "report" / "mhrag_adaptive_questions_300.html"
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
B.BASELINE_METRICS = A.RESULTS / "baseline_metrics.json"
B.BASELINE_RANKINGS = A.RESULTS / "baseline_rankings.jsonl"


if __name__ == "__main__":
    B.run()
