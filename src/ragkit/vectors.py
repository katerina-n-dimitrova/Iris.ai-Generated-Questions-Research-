"""Cosine-similarity helpers shared by every experiment.

All four helpers compute the same normalize-and-dot; they differ in shape
(matrix-vector vs matrix-matrix) and are kept as-is so callers keep their
exact numerical path.
"""

from __future__ import annotations

import numpy as np


def cosine_scores(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    return matrix @ vector / np.maximum(matrix_norm * vector_norm, 1e-12)


def cosine(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    dots = np.einsum("ij,j->i", matrix, vector, optimize=False)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    return dots / np.maximum(norms, 1e-12)


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def normalized_dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    # np.matmul emits spurious overflow warnings with the macOS Accelerate BLAS
    # on these otherwise finite, unit-normalized arrays. einsum is stable and
    # computes the identical pairwise dot product.
    return np.einsum("ik,jk->ij", left, right, optimize=True)
