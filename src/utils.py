"""
Shared utilities: distance metrics and synthetic dataset generation.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def euclidean_distance(point1, point2):
    """
    Euclidean (L2) distance. Good default for low-to-moderate dimensions.
    Becomes less discriminative as dimensionality grows — see DEVELOPMENT_NOTES.md.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def manhattan_distance(point1, point2):
    """
    Manhattan (L1) distance. More stable than L2 in high-dimensional spaces
    (roughly >= 10–50 features) where pairwise L2 distances tend to converge.
    """
    return sum(abs(a - b) for a, b in zip(point1, point2))


DISTANCE_METRICS = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
}


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def make_uniform(n_points, n_dims, low=0.0, high=1.0, seed=None):
    """
    Generate a dataset of points drawn uniformly at random.

    Parameters
    ----------
    n_points : int
    n_dims   : int
    low, high : float  — value range
    seed     : int, optional

    Returns
    -------
    np.ndarray of shape (n_points, n_dims)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_points, n_dims))


def make_clustered(n_points, n_dims, n_clusters=5, cluster_std=0.5, seed=None):
    """
    Generate a dataset with points grouped around random cluster centers.
    Useful for testing how algorithms behave on structured (non-uniform) data.

    Parameters
    ----------
    n_points   : int
    n_dims     : int
    n_clusters : int
    cluster_std : float  — spread of each cluster
    seed       : int, optional

    Returns
    -------
    np.ndarray of shape (n_points, n_dims)
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 10, size=(n_clusters, n_dims))
    points_per_cluster = n_points // n_clusters
    chunks = [
        rng.normal(loc=center, scale=cluster_std, size=(points_per_cluster, n_dims))
        for center in centers
    ]
    return np.vstack(chunks)


def make_gaussian(n_points, n_dims, mean=0.0, std=1.0, seed=None):
    """
    Generate a dataset drawn from a single isotropic Gaussian.

    Parameters
    ----------
    n_points : int
    n_dims   : int
    mean, std : float
    seed     : int, optional

    Returns
    -------
    np.ndarray of shape (n_points, n_dims)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=(n_points, n_dims))


DATASET_GENERATORS = {
    "uniform":   make_uniform,
    "clustered": make_clustered,
    "gaussian":  make_gaussian,
}
