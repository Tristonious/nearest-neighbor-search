"""
Brute-force k-nearest neighbor search.

Guarantees exact results. Scales as O(n * d) per query — every point
in the dataset is compared to the query. Used as the correctness
baseline for accuracy evaluation of approximate methods.

Sorting strategy: compute all distances, then sort with list.sort()
(Timsort, O(n log n)). An alternative is heapq.nsmallest(k, distances)
which runs in O(n log k) and is preferable when k << n, but Timsort is
fast enough in practice for the dataset sizes used here.
"""

from .utils import euclidean_distance


def bf_knn(dataset, query, k, distance_fn=euclidean_distance):
    """
    Return the k nearest neighbors of query in dataset.

    Parameters
    ----------
    dataset : list of list/tuple
    query : list or tuple
    k : int
    distance_fn : callable, optional
        Defaults to euclidean_distance. Pass manhattan_distance for
        high-dimensional experiments.

    Returns
    -------
    list of (float, point)
        k nearest neighbors as (distance, point) pairs, sorted closest-first.
    """
    distances = [(distance_fn(query, point), point) for point in dataset]
    distances.sort(key=lambda x: x[0])
    return distances[:k]
