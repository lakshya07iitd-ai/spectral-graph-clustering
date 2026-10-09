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

        eigen_error = np.linalg.norm(self.L @ v - lam * v)
        norm_error = abs(v @ v - 1)
        orthogonal_error = abs(v @ np.ones(self.n))

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
        _, v = self.fiedler_vector()

        group1 = np.where(v >= threshold)[0]
        group2 = np.where(v < threshold)[0]

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

    def best_cut(self):
        _, v = self.fiedler_vector()
        values = np.sort(v)

        best_value = np.inf
        best_threshold = None
        best_groups = None

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            group1, group2 = self.partition(threshold)

            if len(group1) == 0 or len(group2) == 0:
                continue

            cut = self.cut_value(group1, group2)

            if cut < best_value:
                best_value = cut
                best_threshold = threshold
                best_groups = (group1, group2)

        return best_threshold, best_value, best_groups

    def best_normalized_cut(self):
        _, v = self.fiedler_vector()
        values = np.sort(v)

        best_value = np.inf
        best_threshold = None
        best_groups = None

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            group1, group2 = self.partition(threshold)

            if len(group1) == 0 or len(group2) == 0:
                continue

            ncut = self.normalized_cut(group1, group2)

            if ncut < best_value:
                best_value = ncut
                best_threshold = threshold
                best_groups = (group1, group2)

        return best_threshold, best_value, best_groups


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

    values, _ = graph.eigenpairs()
    lam2, v2 = graph.fiedler_vector()

    print("Vertices:", graph.n)
    print("Edges:", graph.number_of_edges)
    print("Degrees:", graph.degrees.astype(int))
    print("Eigenvalues:", np.round(values, 6))
    print("Fiedler eigenvalue:", lam2)
    print("Fiedler vector:", np.round(v2, 6))
    print("Connected components:", graph.number_of_components())

    x = np.array([-1.0, -0.8, -0.9, -0.6, 0.5, 0.8, 0.7, 1.0])

    print("\nQuadratic form:", graph.quadratic_form(x))
    print("Edge form:", graph.edge_form(x))
    print("Laplacian checks:", graph.check_laplacian())
    print("Fiedler checks:", graph.check_fiedler())

    group1, group2 = graph.partition()

    print("\nPartition at threshold 0:")
    print("Group 1:", group1 + 1)
    print("Group 2:", group2 + 1)
    print("Cut:", graph.cut_value(group1, group2))
    print("Normalized cut:", graph.normalized_cut(group1, group2))

    threshold, cut, groups = graph.best_cut()

    print("\nBest raw cut:")
    print("Threshold:", threshold)
    print("Cut:", cut)
    print("Group 1:", groups[0] + 1)
    print("Group 2:", groups[1] + 1)

    threshold, ncut, groups = graph.best_normalized_cut()

    print("\nBest normalized cut:")
    print("Threshold:", threshold)
    print("Normalized cut:", ncut)
    print("Group 1:", groups[0] + 1)
    print("Group 2:", groups[1] + 1)

    A2 = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 0]
    ])

    graph2 = SpectralGraph(A2)
    threshold, ncut, groups = graph2.best_normalized_cut()

    print("\nSecond graph:")
    print("Fiedler eigenvalue:", graph2.fiedler_vector()[0])
    print("Best normalized cut:", ncut)
    print("Group 1:", groups[0] + 1)
    print("Group 2:", groups[1] + 1)
