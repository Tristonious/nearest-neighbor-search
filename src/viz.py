"""
Visualization helpers for knn-comparison benchmarks.

Produces four figures saved to the working directory:
    fig1_runtime.png         — query runtime vs dimensionality
    fig2_memory.png          — build vs query memory usage
    fig3_kdtree_partition.png — 2D k-d tree space partitioning diagram
    fig4_forest_partitions.png — three randomized trees on the same dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from .kdtree import KDTree
from .ann.early_termination import _build_randomized_tree


# ── Color / marker helpers ────────────────────────────────────────────────────

def _color(name):
    if name == "Brute Force": return "#2E86AB"
    if name == "k-d Tree":    return "#A23B72"
    return "#F18F01"


def _marker(name):
    if name == "Brute Force": return "o"
    if name == "k-d Tree":    return "s"
    for key, m in {"bt=0": "^", "bt=1": "v", "bt=2": "D"}.items():
        if key in name:
            return m
    return "^"


# ── Figure generation ─────────────────────────────────────────────────────────

def generate_visualizations(dimensions, runtime, memory_build, memory_query):
    """
    Generate and save all four benchmark figures.

    Parameters
    ----------
    dimensions : list of int
    runtime : dict  {algo_name: [avg_runtime_per_dim]}
    memory_build : dict  {algo_name: [build_mem_kb_per_dim]}
    memory_query : dict  {algo_name: [query_mem_kb_per_dim]}
    """
    _fig_runtime(dimensions, runtime)
    _fig_memory(dimensions, memory_build, memory_query)
    _fig_kdtree_partition()
    _fig_forest_partitions()


def _fig_runtime(dimensions, runtime):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, vals in runtime.items():
        ax.plot(
            dimensions, vals,
            color=_color(name),
            marker=_marker(name),
            linewidth=2.2 if name in ("Brute Force", "k-d Tree") else 1.2,
            markersize=7,
            alpha=1.0 if name in ("Brute Force", "k-d Tree") else 0.7,
            label=name,
        )
    ax.set_xscale("log")
    ax.set_xlim(1.5, 250)
    ax.set_xticks(dimensions)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xticklabels(dimensions)
    ax.set_xlabel("Dimensionality", fontsize=11)
    ax.set_ylabel("Avg Query Runtime (s)", fontsize=11)
    ax.set_title("Query Runtime vs Dimensionality", fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig("fig1_runtime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_runtime.png")


def _fig_memory(dimensions, memory_build, memory_query):
    # Show only Brute Force, k-d Tree, and Forest ANN bt=0 to keep the chart readable.
    keep = lambda n: n in ("Brute Force", "k-d Tree") or n == "Forest ANN (bt=0)"
    mb = {n: v for n, v in memory_build.items() if keep(n)}
    mq = {n: v for n, v in memory_query.items() if keep(n)}

    x     = np.arange(len(dimensions))
    width = 0.12
    names = list(mb.keys())

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, name in enumerate(names):
        offset = (i - len(names) / 2) * width
        ax.bar(x + offset,         mb[name], width, label=f"{name} (build)", color=_color(name), alpha=0.9)
        ax.bar(x + offset + width, mq[name], width, label=f"{name} (query)", color=_color(name), alpha=0.9, hatch="///")

    ax.set_xlabel("Dimensionality", fontsize=11)
    ax.set_ylabel("Memory Usage (KB)", fontsize=11)
    ax.set_title("Memory Usage: Build vs Query", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig2_memory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig2_memory.png")


def _fig_kdtree_partition():
    pts = [list(np.random.rand(2) * 10) for _ in range(18)]
    tree = KDTree(pts)

    split_colors = ["#2E86AB", "#A23B72", "#F18F01", "#44BBA4"]

    def draw_splits(node, xmin, xmax, ymin, ymax, depth=0):
        if node is None:
            return
        color = split_colors[depth % len(split_colors)]
        lw = max(2.5 - depth * 0.5, 0.8)
        if node.axis == 0:
            x = node.point[0]
            ax.plot([x, x], [ymin, ymax], color=color, linewidth=lw, alpha=0.85)
            draw_splits(node.left,  xmin, x,    ymin, ymax, depth + 1)
            draw_splits(node.right, x,    xmax, ymin, ymax, depth + 1)
        else:
            y = node.point[1]
            ax.plot([xmin, xmax], [y, y], color=color, linewidth=lw, alpha=0.85)
            draw_splits(node.left,  xmin, xmax, ymin, y,    depth + 1)
            draw_splits(node.right, xmin, xmax, y,    ymax, depth + 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    draw_splits(tree.root, 0, 10, 0, 10)

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.scatter(xs, ys, color="#333333", s=55, zorder=5, edgecolors="#ffffff", linewidths=0.8)
    ax.scatter([tree.root.point[0]], [tree.root.point[1]],
               color="#F18F01", s=120, zorder=6, edgecolors="#333333", linewidths=1.5)

    legend_elements = [
        plt.Line2D([0], [0], color=split_colors[i], lw=2.5 - i * 0.5, label=f"Depth {i} split")
        for i in range(4)
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.set_xlabel("X axis", fontsize=10)
    ax.set_ylabel("Y axis", fontsize=10)
    ax.set_title("k-d Tree Space Partitioning (2D)", fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.15, linestyle="--", color="#aaaaaa")
    plt.tight_layout()
    plt.savefig("fig3_kdtree_partition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig3_kdtree_partition.png")


def _fig_forest_partitions():
    pts = [list(np.random.rand(2) * 10) for _ in range(18)]
    split_colors = ["#2E86AB", "#A23B72", "#F18F01", "#44BBA4"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Randomized k-d Tree Forest: 3 Different Space Partitionings",
                 fontsize=13, fontweight="bold")

    for tree_idx, ax in enumerate(axes):
        rand_tree = _build_randomized_tree(pts, candidate_axes=1)

        def draw_splits_ann(node, xmin, xmax, ymin, ymax, depth=0):
            if node is None:
                return
            color = split_colors[depth % len(split_colors)]
            lw = max(2.5 - depth * 0.5, 0.8)
            if node.axis == 0:
                x = node.point[0]
                ax.plot([x, x], [ymin, ymax], color=color, linewidth=lw, alpha=0.85)
                draw_splits_ann(node.left,  xmin, x,    ymin, ymax, depth + 1)
                draw_splits_ann(node.right, x,    xmax, ymin, ymax, depth + 1)
            else:
                y = node.point[1]
                ax.plot([xmin, xmax], [y, y], color=color, linewidth=lw, alpha=0.85)
                draw_splits_ann(node.left,  xmin, xmax, ymin, y,    depth + 1)
                draw_splits_ann(node.right, xmin, xmax, y,    ymax, depth + 1)

        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
        ax.set_title(f"Tree {tree_idx + 1}", fontsize=11)
        draw_splits_ann(rand_tree, 0, 10, 0, 10)

        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.scatter(xs, ys, color="#333333", s=55, zorder=5, edgecolors="#ffffff", linewidths=0.8)
        ax.scatter([rand_tree.point[0]], [rand_tree.point[1]],
                   color="#F18F01", s=120, zorder=6, edgecolors="#333333", linewidths=1.5)
        ax.grid(True, alpha=0.15, linestyle="--", color="#aaaaaa")

    plt.tight_layout()
    plt.savefig("fig4_forest_partitions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig4_forest_partitions.png")
