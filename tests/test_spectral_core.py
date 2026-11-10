import unittest
import numpy as np

from src.spectral_core import SpectralGraph
from experiments.degree_heterogeneity import (
    adjusted_rand_score,
    build_dcsbm_probabilities,
    sample_graph_from_probabilities,
    ordinary_spectral_clustering,
    normalized_spectral_clustering_variants,
    compute_graph_diagnostics,
)
from experiments.regime_interaction_study import (
    regularized_normalized_clustering,
    eigenvector_localization_metrics,
)


class TestSpectralGraphCore(unittest.TestCase):
    def setUp(self):
        self.A = np.array([
            [0, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0]
        ], dtype=float)
        self.graph = SpectralGraph(self.A)

    def test_laplacian_and_fiedler_invariants(self):
        checks = self.graph.check_laplacian()
        self.assertTrue(all(checks.values()))
        f_checks = self.graph.check_fiedler()
        self.assertTrue(f_checks["valid"])

    def test_quadratic_and_edge_form_equivalence(self):
        rng = np.random.default_rng(123)
        for _ in range(5):
            x = rng.normal(size=self.graph.n)
            self.assertAlmostEqual(
                self.graph.quadratic_form(x),
                self.graph.edge_form(x),
                places=10,
            )

    def test_isolated_vertices_and_components(self):
        A_iso = np.zeros((6, 6), dtype=float)
        A_iso[0, 1] = A_iso[1, 0] = 1.0
        A_iso[1, 2] = A_iso[2, 1] = 1.0
        A_iso[3, 4] = A_iso[4, 3] = 1.0
        # vertex 5 is isolated; components are {0,1,2}, {3,4}, {5}
        g_iso = SpectralGraph(A_iso)
        self.assertEqual(g_iso.number_of_components(), 3)
        self.assertTrue(np.all(np.isfinite(g_iso.L_sym)))
        labels, vals, emb, _ = g_iso.spectral_clustering(k=3, seed=42)
        self.assertEqual(len(labels), 6)
        self.assertTrue(np.all(np.isfinite(emb)))

    def test_dcsbm_calibration_invariance(self):
        for gamma in [0.0, 0.45, 0.85]:
            P, labels, theta, info = build_dcsbm_probabilities(
                n_communities=3,
                community_size=10,
                p_in=0.65,
                p_out=0.08,
                gamma=gamma,
                calibrate=True,
            )
            self.assertAlmostEqual(info["expected_m_in"], 87.75, places=5)
            self.assertAlmostEqual(info["expected_m_out"], 24.00, places=5)
            self.assertTrue(np.allclose(P, P.T))
            self.assertTrue(np.allclose(np.diag(P), 0.0))

    def test_clustering_and_localization_pipeline(self):
        P, true_labels, _, _ = build_dcsbm_probabilities(
            n_communities=3,
            community_size=10,
            p_in=0.65,
            p_out=0.08,
            gamma=0.85,
            calibrate=True,
        )
        A = sample_graph_from_probabilities(P, seed=1000)
        g = SpectralGraph(A)
        diag = compute_graph_diagnostics(g, true_labels, k=3)
        self.assertEqual(diag["components"], 1)
        ord_labels, _, _, _ = ordinary_spectral_clustering(g, k=3, seed=42)
        norm_labels, _, _, _, _ = normalized_spectral_clustering_variants(g, k=3, seed=42)
        reg_labels, _ = regularized_normalized_clustering(g, k=3, seed=42)
        loc = eigenvector_localization_metrics(g, k=3)
        self.assertGreaterEqual(adjusted_rand_score(true_labels, norm_labels), 0.8)
        self.assertGreaterEqual(adjusted_rand_score(true_labels, reg_labels), 0.8)
        self.assertGreaterEqual(loc["ord_ipr"], 1.0 / 30.0)


if __name__ == "__main__":
    unittest.main()
