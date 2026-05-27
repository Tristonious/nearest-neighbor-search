"""
run.py — knn-comparison benchmark entry point.

Runs all five algorithms across a configurable set of dimensionalities,
prints results to stdout, appends to results_log.txt, and saves figures
to the working directory.

Usage
-----
    pip install numpy matplotlib
    python run.py

Note on AI assistance
---------------------
The algorithm designs, implementations, experimental methodology, and
analysis in this project are original work. Code structure and
organization were refactored with the assistance of Claude (Anthropic).
See docs/knn_comparison_paper.pdf for the full write-up.
"""

import time
import tracemalloc
import numpy as np

from src import bf_knn, KDTree, ApproximateKDTree, compute_accuracy
from src import make_clustered
from src.viz import generate_visualizations


# ── Parameters ────────────────────────────────────────────────────────────────

DIMENSIONS     = [2, 5, 10, 20, 50, 100, 200]
N_POINTS       = 100_000   # dataset size
N_QUERIES      = 50        # queries per dimensionality
K              = 10        # nearest neighbors

N_TREES        = 20        # forest hyperparameter
CANDIDATE_AXES = 2         # axes sampled per split
BACKTRACKS     = [0, 1, 2] # backtracking budgets to evaluate

# ──────────────────────────────────────────────────────────────────────────────


# ── Logging ───────────────────────────────────────────────────────────────────

log_file = open("results_log.txt", "a")


def log(*args, **kwargs):
    """Print to stdout and append to results_log.txt."""
    print(*args, **kwargs)
    print(*args, **{k: v for k, v in kwargs.items() if k != "file"}, file=log_file, flush=True)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmarks():
    log(f"\n{'=' * 85}\nRun started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 85}\n")
    log(f"{'Dim':<6} {'Algorithm':<20} {'Avg Runtime (s)':<20} {'Build Mem (KB)':<15} {'Query Mem (KB)':<15} {'Accuracy'}")
    log("-" * 85)

    results = {
        "Brute Force": {"runtime": [], "memory_query": [], "memory_build": [], "accuracy": []},
        "k-d Tree":    {"runtime": [], "memory_query": [], "memory_build": [], "accuracy": []},
    }
    for bt in BACKTRACKS:
        results[f"Forest ANN (bt={bt})"] = {"runtime": [], "memory_query": [], "memory_build": [], "accuracy": []}

    for dim in DIMENSIONS:
        dataset = make_clustered(N_POINTS, dim)
        queries = make_clustered(N_QUERIES, dim)

        # Ground truth from brute force
        bf_results    = [bf_knn(dataset, q, K) for q in queries]
        bf_neighbors  = [[p for _, p in r] for r in bf_results]

        # Build trees once per dimension; measure build memory separately.
        tracemalloc.start()
        kd_tree = KDTree(dataset)
        _, kd_build_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        kd_build_mem = kd_build_peak / 1024

        tracemalloc.start()
        forest = ApproximateKDTree(dataset, n_trees=N_TREES, candidate_axes=CANDIDATE_AXES)
        _, forest_build_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        forest_build_mem = forest_build_peak / 1024

        # Define query callables (build cost excluded)
        def run_bf(q):
            return [p for _, p in bf_knn(dataset, q, K)]

        def run_kd(q):
            return [p for _, p in kd_tree.query(q, K)]

        named_algos = [
            ("Brute Force", run_bf),
            ("k-d Tree",    run_kd),
        ]
        for bt in BACKTRACKS:
            def make_ann(backtracks):
                def run_ann(q):
                    return [p for _, p in forest.query(q, K, backtracks=backtracks)]
                return run_ann
            named_algos.append((f"Forest ANN (bt={bt})", make_ann(bt)))

        # Run each algorithm across all queries
        for name, algo in named_algos:
            runtimes, memories, accuracies = [], [], []

            for i, query in enumerate(queries):
                # Runtime
                t0 = time.perf_counter()
                result = algo(query)
                runtimes.append(time.perf_counter() - t0)

                # Query memory
                tracemalloc.start()
                _ = algo(query)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memories.append(peak / 1024)

                # Accuracy vs brute force ground truth
                accuracies.append(compute_accuracy(bf_neighbors[i], result))

            avg_rt  = np.mean(runtimes)
            avg_mem = np.mean(memories)
            avg_acc = np.mean(accuracies)

            results[name]["runtime"].append(avg_rt)
            results[name]["memory_query"].append(avg_mem)
            results[name]["accuracy"].append(avg_acc)

            if name == "k-d Tree":
                results[name]["memory_build"].append(kd_build_mem)
            elif "Forest ANN" in name:
                results[name]["memory_build"].append(forest_build_mem)
            else:
                results[name]["memory_build"].append(0)

            log(f"{dim:<6} {name:<20} {avg_rt:<20.6f} {results[name]['memory_build'][-1]:<15.2f} {avg_mem:<15.2f} {avg_acc:.4f}")

        log()  # blank line between dimensionalities

    # Generate figures
    runtime_viz      = {name: results[name]["runtime"]     for name in results}
    memory_build_viz = {name: results[name]["memory_build"] for name in results}
    memory_query_viz = {name: results[name]["memory_query"] for name in results}

    generate_visualizations(DIMENSIONS, runtime_viz, memory_build_viz, memory_query_viz)
    log_file.close()


if __name__ == "__main__":
    run_benchmarks()
