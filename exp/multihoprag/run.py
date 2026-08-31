#!/usr/bin/env python3
"""Run the five MultiHop-RAG conditions in their required order.

The condition scripts share one metrics.json and one report, so order matters:
`baseline` computes condition 1 and seeds the chunk/query vectors,
`adaptive_questions` writes conditions 1-3, `chunk_article_questions` appends
condition 4, and `chunk_article_unbounded` appends condition 5 and renders the
final five-row report.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline
import adaptive_questions
import chunk_article_questions
import chunk_article_unbounded


def main() -> None:
    if not (
        baseline.METRICS.exists()
        and baseline.CHUNK_VECTORS.exists()
        and baseline.QUERY_VECTORS.exists()
    ):
        baseline.run()
    adaptive_questions.run()
    chunk_article_questions.run()
    chunk_article_unbounded.run()


if __name__ == "__main__":
    main()
