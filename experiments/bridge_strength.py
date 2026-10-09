
import numpy as np
from src.spectral_core import SpectralGraph


def make_graph(bridge_edges):
    A = np.zeros((8, 8))

    communities = [
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ]

    for group in communities:
        for i in group:
            for j in group:
                if i != j:
                    A[i, j] = 1

    for i, j in bridge_edges:
        A[i, j] = 1
        A[j, i] = 1

    return A


bridge_cases = [
    [],
    [(3, 4)],
    [(3, 4), (2, 5)],
    [(3, 4), (2, 5), (1, 6)],
    [(3, 4), (2, 5), (1, 6), (0, 7)]
]

print("Bridges | Fiedler value | Normalized cut")

for bridges in bridge_cases:
    graph = SpectralGraph(make_graph(bridges))
    lam2, _ = graph.fiedler_vector()

    _, ncut, groups = graph.best_normalized_cut()

    print(f"{len(bridges):7d} | {lam2:13.6f} | {ncut:.6f}")
    print("Groups:", groups[0] + 1, "|", groups[1] + 1)
