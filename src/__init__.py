"""
knn-comparison — src package

Public API. Import algorithms and utilities from here rather than
reaching into submodules directly.

    from src import bf_knn, KDTree, ApproximateKDTree
    from src import make_clustered, compute_accuracy, euclidean_distance
"""

from .brute_force import bf_knn
from .kdtree import KDTree
from .ann import ApproximateKDTree
from .utils import (
    euclidean_distance,
    manhattan_distance,
    DISTANCE_METRICS,
    make_uniform,
    make_clustered,
    make_gaussian,
    DATASET_GENERATORS,
    compute_accuracy,
)
