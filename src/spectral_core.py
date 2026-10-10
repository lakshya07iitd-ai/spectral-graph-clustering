
import numpy as np


class SpectralGraph:
    def __init__(self, adjacency):
        self.A = np.array(adjacency, dtype=float)
        self._check_adjacency()

        self.n = self.A.shape[0]
        self.degrees = self.A.sum(axis=1)
        self.D = np.diag(self.degrees)
        self.L = self.D - self.A

        self.inv_sqrt_degrees = np.zeros(self.n)
        positive = self.degrees > 0
        self.inv_sqrt_degrees[positive] = (
            1 / np.sqrt(self.degrees[positive])
        )

        self.D_inv_sqrt = np.diag(self.inv_sqrt_degrees)
        self.L_sym = self.D_inv_sqrt @ self.L @ self.D_inv_sqrt

    def _check_adjacency(self):
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if not np.all(np.isfinite(self.A)):
            raise ValueError("Adjacency matrix must contain finite values")
        if not np.allclose(self.A, self.A.T):
            raise ValueError("Adjacency matrix must be symmetric")
        if np.any(self.A < 0):
            raise ValueError("Edge weights must be non-negative")
        if not np.allclose(np.diag(self.A), 0):
            raise ValueError("Self-loops are not supported")

    @property
    def number_of_edges(self):
        return float(self.A.sum() / 2)

    def eigenpairs(self):
        return np.linalg.eigh(self.L)

    def normalized_eigenpairs(self):
        return np.linalg.eigh(self.L_sym)

    def fiedler_vector(self):
        values, vectors = self.eigenpairs()
        return values[1], vectors[:, 1]

    def normalized_fiedler_vector(self):
        values, vectors = self.normalized_eigenpairs()
        vector = self.D_inv_sqrt @ vectors[:, 1]
        return values[1], vector

    def quadratic_form(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise ValueError(f"x must have {self.n} elements")
        return float(x @ self.L @ x)

    def edge_form(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise ValueError(f"x must have {self.n} elements")

        diff = x[:, None] - x[None, :]
        return float(0.5 * np.sum(self.A * diff**2))

    def number_of_components(self, tolerance=1e-10):
        values, _ = self.eigenpairs()
        return int(np.sum(np.abs(values) < tolerance))

    def check_laplacian(self):
        return {
            "symmetric": np.allclose(self.L, self.L.T),
            "L1_zero": np.allclose(self.L @ np.ones(self.n), 0),
            "normalized_symmetric": np.allclose(
                self.L_sym, self.L_sym.T
            ),
            "normalized_finite": np.all(np.isfinite(self.L_sym))
        }

    def check_fiedler(self, tolerance=1e-8):
        values, vectors = self.eigenpairs()
        lam = values[1]
        vector = vectors[:, 1]

        eigen_error = np.linalg.norm(self.L @ vector - lam * vector)
        norm_error = abs(vector @ vector - 1)
        orthogonal_error = abs(vector @ np.ones(self.n))

        return {
            "eigen_error": eigen_error,
            "norm_error": norm_error,
            "orthogonal_error": orthogonal_error,
            "valid": (
                eigen_error < tolerance
                and norm_error < tolerance
                and orthogonal_error < tolerance
            )
        }

    def partition(self, threshold=0):
        _, vector = self.fiedler_vector()
        return self._split(vector, threshold)

    def normalized_partition(self, threshold=0):
        _, vector = self.normalized_fiedler_vector()
        return self._split(vector, threshold)

    @staticmethod
    def _split(vector, threshold):
        group1 = np.where(vector >= threshold)[0]
        group2 = np.where(vector < threshold)[0]
        return group1, group2

    def cut_value(self, group1, group2):
        return float(self.A[np.ix_(group1, group2)].sum())

    def volume(self, group):
        return float(self.degrees[group].sum())

    def normalized_cut(self, group1, group2):
        vol1 = self.volume(group1)
        vol2 = self.volume(group2)

        if vol1 == 0 or vol2 == 0:
            return np.inf

        cut = self.cut_value(group1, group2)
        return cut / vol1 + cut / vol2

    def _best_threshold(self, vector, objective):
        values = np.sort(np.unique(vector))

        best_value = np.inf
        best_threshold = None
        best_groups = None

        for left, right in zip(values[:-1], values[1:]):
            threshold = (left + right) / 2
            group1, group2 = self._split(vector, threshold)

            if len(group1) == 0 or len(group2) == 0:
                continue

            score = objective(group1, group2)

            if score < best_value:
                best_value = score
                best_threshold = threshold
                best_groups = (group1, group2)

        return best_threshold, best_value, best_groups

    def best_cut(self):
        _, vector = self.fiedler_vector()
        return self._best_threshold(
            vector,
            lambda g1, g2: self.cut_value(g1, g2)
        )

    def best_normalized_cut(self):
        _, vector = self.normalized_fiedler_vector()
        return self._best_threshold(vector, self.normalized_cut)


if __name__ == "__main__":
    A = np.array([
        [0, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 0]
    ], dtype=float)

    graph = SpectralGraph(A)

    print("Vertices:", graph.n)
    print("Edges:", graph.number_of_edges)
    print("Degrees:", graph.degrees.astype(int))
    print("Connected components:", graph.number_of_components())
    print("Matrix checks:", graph.check_laplacian())
    print("Fiedler checks:", graph.check_fiedler())

    lam, vector = graph.fiedler_vector()
    print("\nOrdinary Fiedler eigenvalue:", round(lam, 6))
    print("Ordinary partition:", [g + 1 for g in graph.partition()])

    lam_sym, vector_sym = graph.normalized_fiedler_vector()
    print("\nNormalized eigenvalue:", round(lam_sym, 6))
    print("Normalized partition:", [
        g + 1 for g in graph.normalized_partition()
    ])

    threshold, score, groups = graph.best_cut()
    print("\nBest raw cut:", score)
    print("Groups:", groups[0] + 1, groups[1] + 1)

    threshold, score, groups = graph.best_normalized_cut()
    print("\nBest normalized cut:", score)
    print("Groups:", groups[0] + 1, groups[1] + 1)

