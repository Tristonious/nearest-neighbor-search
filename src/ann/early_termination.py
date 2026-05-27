"""
Approximate nearest-neighbor search via a randomized k-d tree forest
with a configurable backtracking budget.

Each tree is built by splitting on the highest-spread axis among a
random sample of candidate_axes axes at each level. Different trees
partition the space differently; querying all of them and merging
results gives better approximate coverage than any single tree.

At query time, backtracking controls how many times the algorithm is
allowed to cross a splitting hyperplane to check the far subtree.
bt=0 is a pure greedy DFS (fastest, least accurate); higher values
recover more true neighbors at the cost of additional traversal.

Conceptually similar to Spotify's ANNOY library.

References
----------
J. Lu and H. Gweon. 2019. Random k conditional nearest neighbor.
    Pattern Recognition 96.
E. Bernhardsson. 2015. Annoy. https://github.com/spotify/annoy
"""

import random
from ..utils import euclidean_distance
from ..kdtree import KDNode


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_randomized_tree(points, depth=0, candidate_axes=2):
    """
    Build one randomized k-d tree.

    At each level, sample `candidate_axes` axes at random and split on
    whichever has the highest spread. This produces a different
    partitioning than a standard k-d tree and varies across trees in
    the forest.
    """
    if not points:
        return None

    ndim = len(points[0])
    sampled = random.sample(range(ndim), min(candidate_axes, ndim))
    axis = max(
        sampled,
        key=lambda a: max(p[a] for p in points) - min(p[a] for p in points)
    )

    points_sorted = sorted(points, key=lambda p: p[axis])
    median_idx = len(points_sorted) // 2

    return KDNode(
        point=points_sorted[median_idx],
        axis=axis,
        left=_build_randomized_tree(points_sorted[:median_idx], depth + 1, candidate_axes),
        right=_build_randomized_tree(points_sorted[median_idx + 1:], depth + 1, candidate_axes),
    )


def _query_backtrack(node, query, k, best_k, backtracking, distance_fn):
    """
    DFS query with a backtracking budget.

    Each time the algorithm would cross to the far subtree it consumes
    one unit of backtracking. When the budget reaches zero, far-side
    exploration stops entirely.
    """
    if node is None:
        return

    d = distance_fn(query, node.point)

    if len(best_k) < k:
        best_k.append((d, node.point))
        best_k.sort(key=lambda x: x[0])
    elif d < best_k[-1][0]:
        best_k[-1] = (d, node.point)
        best_k.sort(key=lambda x: x[0])

    diff = query[node.axis] - node.point[node.axis]
    near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)

    _query_backtrack(near, query, k, best_k, backtracking, distance_fn)

    if backtracking > 0 and (len(best_k) < k or abs(diff) < best_k[-1][0]):
        _query_backtrack(far, query, k, best_k, backtracking - 1, distance_fn)


# ── Public API ────────────────────────────────────────────────────────────────

class ApproximateKDTree:
    """
    Randomized k-d tree forest for approximate nearest-neighbor search.

    Usage
    -----
    ann = ApproximateKDTree(dataset, n_trees=20, candidate_axes=2)
    neighbors = ann.query(query_point, k=10, backtracks=2)
    # returns list of (distance, point) sorted closest-first
    """

    def __init__(self, points, n_trees=20, candidate_axes=2, distance_fn=euclidean_distance):
        """
        Build the forest.

        Parameters
        ----------
        points : list of list/tuple
        n_trees : int
            Number of randomized trees. More trees = higher accuracy,
            higher build cost.
        candidate_axes : int
            Axes sampled per split. Higher = less randomness, closer to
            exact k-d tree behavior.
        distance_fn : callable
        """
        self.distance_fn = distance_fn
        self.forest = [
            _build_randomized_tree(points, candidate_axes=candidate_axes)
            for _ in range(n_trees)
        ]

    def query(self, query_point, k, backtracks=2):
        """
        Return approximate k nearest neighbors.

        Parameters
        ----------
        query_point : list or tuple
        k : int
        backtracks : int
            Per-tree backtracking budget. bt=0 is pure greedy DFS.

        Returns
        -------
        list of (float, point) sorted closest-first, length <= k
        """
        candidates = []
        for tree in self.forest:
            tree_best = []
            _query_backtrack(tree, query_point, k, tree_best, backtracks, self.distance_fn)
            candidates.extend(tree_best)

        # Merge, deduplicate, and return top k.
        candidates.sort(key=lambda x: x[0])
        seen, deduped = set(), []
        for d, p in candidates:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                deduped.append((d, p))

        return deduped[:k]
