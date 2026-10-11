
import numpy as np

from src.spectral_core import SpectralGraph


def build_graph():
    adjacency = np.zeros((9, 9))

    communities = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]

    for group in communities:
        for i in group:
            for j in group:
                if i != j:
                    adjacency[i, j] = 1

    adjacency[2, 3] = adjacency[3, 2] = 0.2
    adjacency[5, 6] = adjacency[6, 5] = 0.2

    return adjacency


def main():
    graph = SpectralGraph(build_graph())

    labels, eigenvalues, embedding, centers = (
        graph.spectral_clustering(k=3)
    )

    print("Number of vertices:", graph.n)
    print("Connected components:", graph.number_of_components())
    print("Selected eigenvalues:", np.round(eigenvalues, 4))
    print("Cluster assignments:", labels)
    print("Embedding shape:", embedding.shape)
    print("Cluster sizes:", np.bincount(labels, minlength=3))


if __name__ == "__main__":
    main()
