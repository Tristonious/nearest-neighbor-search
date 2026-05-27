# Development Notes

Scratch notes, open questions, and implementation decisions made during development.
Not intended to be polished — just a running log.

---

## Distance Metric Choice: Euclidean vs. Manhattan

The current implementation uses **Euclidean (L2) distance**, but Manhattan (L1) is worth considering.

After looking into it: this is not a cut-and-dry answer.

- **Euclidean** — less numerically stable in high-dimensional spaces, but works better for small/low-dimensional datasets. Distances between points stay meaningful.
- **Manhattan** — better suited for higher-dimensional data (roughly 10–50+ features). As dimensionality grows, all pairwise L2 distances tend to converge, making them less useful for ranking neighbors. L1 degrades more gracefully.

**Decision:** For the scope of this project (comparing behavior across dimensionalities), it makes sense to implement *both* and let the caller choose. The `distance_fn` parameter in `bf_knn` handles this — pass `manhattan_distance` for high-dimensional experiments.

---

## Brute-Force (`bf_knn`) — Sorting

Open question from initial implementation: *what's the best sorting strategy for inserting a new distance into the results list?*

Current approach: compute all distances first, then sort the full list with `list.sort()`.

- `list.sort()` uses Timsort — O(n log n), stable, very fast in practice.
- Alternative: use `heapq.nsmallest(k, distances)` — O(n log k), better when k << n.
- Alternative: maintain a max-heap of size k as we iterate — avoids sorting the full list at all.

For now `list.sort()` is fine since we're in exploratory/comparison mode. If brute-force becomes a bottleneck in experiments, switch to `heapq.nsmallest`.

The original `pop` idea in the pseudocomment (pop x from the sorted list) — didn't end up using it; slicing with `[:k]` is cleaner and doesn't mutate the list.

---

## k-D Tree — TODO

Still need to implement:

- `KDTree.build(points)` — recursive construction, splitting on the axis of highest variance at each level
- `KDTree.query(query_point, k)` — recursive descent with backtracking and branch pruning

Key implementation notes to keep in mind:
- Axis selection: splitting on the dimension with the highest spread (variance or range) tends to produce more balanced trees.
- Pruning condition: prune a subtree if the distance from the query to the splitting hyperplane is greater than the current k-th best distance found.
- Degrades toward brute-force as dimensionality increases — this is expected and part of what the experiments will quantify.

---

## ANN (Early Termination k-D Tree) — TODO

Need to add an `epsilon` parameter: stop backtracking once the best candidate is within `(1 + epsilon)` of the true nearest neighbor distance.

This is sometimes called a "defeatist" search — intentionally giving up on backtracking past a certain point. Simpler than graph-based ANN but still a useful baseline.

---

## Experiments — Rough Plan

Generate synthetic datasets at d = 2, 5, 10, 20, 50, 100 dimensions.
For each: measure brute-force query time, k-d tree query time, ANN query time, ANN recall@k.

Hypothesis: k-d tree beats brute-force up to ~20 dimensions, then converges; ANN stays competitive throughout.
