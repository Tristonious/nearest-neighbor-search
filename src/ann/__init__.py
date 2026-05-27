"""
ann — Approximate Nearest Neighbor subpackage.

Current implementations
-----------------------
ApproximateKDTree : randomized k-d tree forest with backtracking budget

Planned
-------
KNNGraph     : graph-based ANN (Hajebi et al. 2011)
HNSW         : hierarchical navigable small world (Malkov & Yashunin 2018)
"""

from .early_termination import ApproximateKDTree
