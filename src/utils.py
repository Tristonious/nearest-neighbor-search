"""
Shared utilities for knn-comparison.

Provides distance metrics, synthetic dataset generators, and the
accuracy helper used across all benchmark algorithms.
"""

import math
import numpy as np


# ── Distance metrics ──────────────────────────────────────────────────────────

def euclidean_distance(p1, p2):
    """L2 distance. Stable for low-to-moderate dimensionality."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def manhattan_distance(p1, p2):
    """L1 distance. Degrades more gracefully at high dimensionality than L2."""
    return sum(abs(a - b) for a, b in zip(p1, p2))


DISTANCE_METRICS = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
}


# ── Dataset generators ────────────────────────────────────────────────────────

def make_uniform(n, dim):
    """n points drawn uniformly from [0, 1]^dim."""
    return [list(np.random.rand(dim)) for _ in range(n)]


def make_clustered(n, dim, n_clusters=10):
    """
    n points drawn from n_clusters Gaussian clusters with centers in [0, 10]^dim
    and std=1.5. Produces realistic non-uniform structure for benchmarking.
    """
    centers = np.random.rand(n_clusters, dim) * 10
    dataset = []
    for _ in range(n):
        center = centers[np.random.randint(n_clusters)]
        point = center + np.random.randn(dim) * 1.5
        dataset.append(list(point))
    return dataset


def make_gaussian(n, dim):
    """n points drawn from a zero-mean unit-variance Gaussian."""
    return [list(np.random.randn(dim)) for _ in range(n)]


DATASET_GENERATORS = {
    "uniform":   make_uniform,
    "clustered": make_clustered,
    "gaussian":  make_gaussian,
}


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_accuracy(true_neighbors, approx_neighbors):
    """
    Fraction of true k-NN returned by an approximate search.

    Parameters
    ----------
    true_neighbors : list of point (list/tuple)
    approx_neighbors : list of point (list/tuple)

    Returns
    -------
    float in [0, 1]
    """
    true_set = set(tuple(p) for p in true_neighbors)
    approx_set = set(tuple(p) for p in approx_neighbors)
    if not true_set:
        return 1.0
    return len(true_set & approx_set) / len(true_set)
