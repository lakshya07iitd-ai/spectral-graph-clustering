import numpy as np


class SpectralGraph:
    def __init__(self, adjacency):
        self.A = np.array(adjacency, dtype=float)
        self._check_adjacency()

        self.n = self.A.shape[0]
        self.D = np.diag(self.A.sum(axis=1))
        self.L = self.D - self.A

    def _check_adjacency(self):
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Adjacency matrix must be square")

        if not np.allclose(self.A, self.A.T):
            raise ValueError("Adjacency matrix must be symmetric")

        if np.any(self.A < 0):
            raise ValueError("Edge weights must be non-negative")

        if not np.allclose(np.diag(self.A), 0):
            raise ValueError("Self-loops are not supported")

    @property
    def degrees(self):
        return self.A.sum(axis=1)

    @property
    def number_of_edges(self):
        return self.A.sum() / 2

    def eigenpairs(self):
        return np.linalg.eigh(self.L)

    def fiedler_vector(self):
        values, vectors = self.eigenpairs()
        return values[1], vectors[:, 1]

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
        ones = np.ones(self.n)

        return {
            "symmetric": np.allclose(self.L, self.L.T),
            "L1_zero": np.allclose(self.L @ ones, 0)
        }

    def check_fiedler(self, tolerance=1e-8):
        values, vectors = self.eigenpairs()

        lam = values[1]
        v = vectors[:, 1]
        ones = np.ones(self.n)

        eigen_error = np.linalg.norm(self.L @ v - lam * v)
        norm_error = abs(v @ v - 1)
        orthogonal_error = abs(v @ ones)

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
    ])

    graph = SpectralGraph(A)

    values, vectors = graph.eigenpairs()
    lam2, v2 = graph.fiedler_vector()

    print("Number of vertices:", graph.n)
    print("Number of edges:", graph.number_of_edges)
    print("Degrees:", graph.degrees.astype(int))

    print("\nLaplacian:")
    print(graph.L)

    print("\nEigenvalues:")
    print(np.round(values, 6))

    print("\nFiedler eigenvalue:")
    print(lam2)

    print("\nFiedler vector:")
    print(np.round(v2, 6))

    x = np.array([
        -1.0, -0.8, -0.9, -0.6,
         0.5,  0.8,  0.7,  1.0
    ])

    matrix_value = graph.quadratic_form(x)
    edge_value = graph.edge_form(x)

    print("\nQuadratic form:")
    print("x^T L x =", matrix_value)
    print("Edge form =", edge_value)
    print("Difference =", abs(matrix_value - edge_value))

    print("\nLaplacian checks:")
    print(graph.check_laplacian())

    print("\nFiedler checks:")
    print(graph.check_fiedler())

    print("\nNumber of components:")
    print(graph.number_of_components())
