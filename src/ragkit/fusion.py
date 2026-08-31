"""Reciprocal-rank fusion shared by every experiment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

RRF_K = 60


def rrf_merge(rankings: Iterable[list[str]], k: int = RRF_K) -> list[str]:
    """Fuse rankings of chunk ids; ties break lexicographically by id."""
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] += 1.0 / (k + rank)
    return [item[0] for item in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
