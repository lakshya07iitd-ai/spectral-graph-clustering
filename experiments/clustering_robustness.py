
import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.spectral_core import SpectralGraph


def build_graph(p_in, p_out, seed):
    rng = np.random.default_rng(seed)
    n_communities = 3
    community_size = 10
    n = n_communities * community_size

    labels = np.repeat(np.arange(n_communities), community_size)
    adjacency = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            same_community = labels[i] == labels[j]
            probability = p_in if same_community else p_out

            if rng.random() < probability:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    return adjacency, labels


def main():
    settings = [
        (0.8, 0.02),
        (0.6, 0.10),
        (0.4, 0.20),
        (0.3, 0.25),
        (0.25, 0.25),
    ]

    seeds = range(10)

    print("Spectral Clustering Robustness Experiment")
    print("-" * 72)
    print(
        f"{'p_in':>7} {'p_out':>7} "
        f"{'Mean ARI':>12} {'Std ARI':>12} {'Mean ARI baseline':>19}"
    )

    for p_in, p_out in settings:
        scores = []
        baseline_scores = []

        for seed in seeds:
            adjacency, true_labels = build_graph(p_in, p_out, seed)
            graph = SpectralGraph(adjacency)

            predicted, eigenvalues, embedding, centers = (
                graph.spectral_clustering(k=3, seed=42)
            )

            scores.append(
                adjusted_rand_score(true_labels, predicted)
            )

            random_labels = np.random.default_rng(seed + 1000).integers(
                0, 3, size=len(true_labels)
            )
            baseline_scores.append(
                adjusted_rand_score(true_labels, random_labels)
            )

        print(
            f"{p_in:7.2f} {p_out:7.2f} "
            f"{np.mean(scores):12.4f} {np.std(scores):12.4f} "
            f"{np.mean(baseline_scores):19.4f}"
        )


if __name__ == "__main__":
    main()

