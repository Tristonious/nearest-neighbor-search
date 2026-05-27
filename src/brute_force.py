"""
Brute-force k-nearest neighbor search.

Guarantees exact results. Scales as O(n * d) per query — every point
in the dataset is compared to the query. Useful as a correctness
baseline and for small datasets.
"""

from .utils import euclidean_distance


def bf_knn(dataset, query, k, distance_fn=euclidean_distance):
    """
    Return the k nearest neighbors of query in dataset.

    Parameters
    ----------
    dataset : list of tuple/list
    query : tuple or list
    k : int
    distance_fn : callable, optional
        Defaults to euclidean_distance. Pass manhattan_distance for
        high-dimensional data.

    Returns
    -------
    list of (float, point)
        k nearest neighbors as (distance, point) pairs, sorted closest-first.
    """
    distances = [(distance_fn(query, point), point) for point in dataset]
    distances.sort(key=lambda x: x[0])
    return distances[:k]
