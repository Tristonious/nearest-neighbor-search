"""
ann — Approximate Nearest Neighbor subpackage

Current implementations
-----------------------
ApproximateKDTree : k-d tree with early termination (epsilon-based pruning)

Planned
-------
KNNGraph         : graph-based ANN (Hajebi et al. 2011)
RandomizedANN    : randomized search (Lu & Gweon 2019)
ClusterANN       : cluster-based adaptive indexing (Kazakovtsev et al. 2025)
"""

from .early_termination import ApproximateKDTree
