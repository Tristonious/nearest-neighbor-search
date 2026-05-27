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

## k-D Tree — Implemented

`KDTree` is implemented in `src/kdtree.py` as a class wrapping `KDNode`.

- **Build:** splits on the axis of highest spread (range) at each level, placing the median point at the node. Recursive construction, left subtree gets points below the median, right gets points above.
- **Query:** recursive descent with backtracking. Always visits the near side first. Only crosses to the far side if the distance to the splitting hyperplane is less than the current k-th best distance — otherwise prunes the subtree entirely.
- Degrades toward brute-force at high dimensionality as expected; crossover observed around d=20 in experiments.

---

## ANN (Randomized k-d Tree Forest) — Implemented

Implemented in `src/ann/early_termination.py` as `ApproximateKDTree`.

Went with a **backtracking budget** rather than an epsilon parameter. At each node the algorithm descends the near side; the `backtracks` parameter controls how many times it is allowed to cross to the far side. bt=0 is a pure greedy DFS, bt=2 recovers most true neighbors at low-to-moderate dimensionality.

Built as a forest of 20 randomized trees (candidate_axes=2). Each tree partitions the space differently by sampling a random subset of axes at each split rather than always picking the globally best one. Results from all trees are merged and deduplicated before returning the top k.

Epsilon-based early termination (stop when best candidate is within (1 + ε) of the true NN) is still a viable future extension.

---

## Experiments — Completed

Results across three runs logged in `results_log.txt`. Tested d = 2, 5, 10, 20, 50, 100, 200 on 100,000-point synthetic clustered datasets, 50 queries per dimensionality, k=10.

Hypothesis held: k-d tree beats brute-force up to ~d=20, converges around d=50, and marginally exceeds brute-force at d=200. All ANN variants (bt=0,1,2) remain faster than both exact methods at every dimensionality tested. Accuracy drops sharply with dimension for bt=0; bt=2 holds up reasonably well through d=20.
