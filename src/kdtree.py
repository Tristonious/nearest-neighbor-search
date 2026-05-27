"""
k-d tree for exact nearest-neighbor search.

Introduced by Bentley (1975). Recursively partitions space along
coordinate axes, allowing large regions of the search space to be
pruned during queries. Runs in O(log n) average time for low-to-moderate
dimensions; degrades toward brute-force as dimensionality increases.

Reference
---------
J. L. Bentley. 1975. Multidimensional binary search trees used for
associative searching. Communications of the ACM 18, 9, 509-517.
"""

from .utils import euclidean_distance


class KDNode:
    """Single node in a k-d tree."""

    def __init__(self, point, axis, left=None, right=None):
        self.point = point   # median point stored at this node
        self.axis  = axis    # dimension we split on
        self.left  = left    # subtree with points < median on this axis
        self.right = right   # subtree with points >= median on this axis


class KDTree:
    """
    Axis-aligned binary space-partitioning tree.

    Splits at each level on the axis of highest spread (range), placing
    the median point at the node. Query uses recursive descent with
    backtracking: the far subtree is pruned when the distance to the
    splitting hyperplane exceeds the current k-th best distance.

    Usage
    -----
    tree = KDTree(dataset)
    neighbors = tree.query(query_point, k=10)
    # returns list of (distance, point) sorted closest-first
    """

    def __init__(self, points, distance_fn=euclidean_distance):
        self.distance_fn = distance_fn
        self.root = self._build(points)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self, points, depth=0):
        if not points:
            return None

        ndim = len(points[0])

        # Split on the axis with the largest spread so the most
        # discriminating splits land near the root.
        axis = max(
            range(ndim),
            key=lambda a: max(p[a] for p in points) - min(p[a] for p in points)
        )

        points_sorted = sorted(points, key=lambda p: p[axis])
        median_idx = len(points_sorted) // 2

        return KDNode(
            point=points_sorted[median_idx],
            axis=axis,
            left=self._build(points_sorted[:median_idx], depth + 1),
            right=self._build(points_sorted[median_idx + 1:], depth + 1),
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def _query(self, node, query, k, best_k):
        if node is None:
            return

        d = self.distance_fn(query, node.point)

        if len(best_k) < k:
            best_k.append((d, node.point))
            best_k.sort(key=lambda x: x[0])
        elif d < best_k[-1][0]:
            best_k[-1] = (d, node.point)
            best_k.sort(key=lambda x: x[0])

        # Decide near/far side relative to the splitting hyperplane.
        diff = query[node.axis] - node.point[node.axis]
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)

        # Always descend the near side first.
        self._query(near, query, k, best_k)

        # Only descend the far side if it could contain a closer neighbor:
        # the distance to the hyperplane is abs(diff); if that exceeds the
        # current worst neighbor in best_k, nothing on the far side can win.
        if len(best_k) < k or abs(diff) < best_k[-1][0]:
            self._query(far, query, k, best_k)

    def query(self, query_point, k):
        """
        Return the k nearest neighbors of query_point.

        Parameters
        ----------
        query_point : list or tuple
        k : int

        Returns
        -------
        list of (float, point) sorted closest-first
        """
        best_k = []
        self._query(self.root, query_point, k, best_k)
        return best_k
