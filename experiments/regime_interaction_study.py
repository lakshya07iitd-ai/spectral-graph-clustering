import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.degree_heterogeneity import (
    adjusted_rand_score,
    build_dcsbm_probabilities,
    sample_graph_from_probabilities,
    ordinary_spectral_clustering,
    normalized_spectral_clustering_variants,
    compute_graph_diagnostics,
    paired_statistics,
)
from src.spectral_core import SpectralGraph


def regularized_normalized_clustering(graph, k=3, tau=None, max_iter=100, seed=42):
    degrees = graph.degrees
    if tau is None:
        tau = float(np.mean(degrees)) * 0.25
    reg_deg = degrees + tau
    inv_sqrt = np.where(reg_deg > 0, 1.0 / np.sqrt(reg_deg), 0.0)
    D_reg_inv_sqrt = np.diag(inv_sqrt)
    L_reg = np.eye(graph.n) - D_reg_inv_sqrt @ graph.A @ D_reg_inv_sqrt
    L_reg = 0.5 * (L_reg + L_reg.T)
    vals, vecs = np.linalg.eigh(L_reg)
    emb = vecs[:, :k].copy()
    norms = np.linalg.norm(emb, axis=1)
    nz = norms > 1e-12
    emb[nz] /= norms[nz, None]
    labels, _ = graph._kmeans(emb, k, max_iter=max_iter, seed=seed)
    return labels, vals


def eigenvector_localization_metrics(graph, k=3):
    ord_vals, ord_vecs = graph.eigenpairs()
    sym_vals, sym_vecs = graph.normalized_eigenpairs()

    # Participation ratio / Inverse Participation Ratio (IPR) of nontrivial eigenvectors 1..k-1
    # IPR(v) = sum(v_i^4); for uniform vector IPR = 1/n = 1/30 = 0.0333; for localized delta vector IPR = 1.0
    ord_ipr = float(np.mean(np.sum(ord_vecs[:, 1:k] ** 4, axis=0)))
    sym_ipr = float(np.mean(np.sum(sym_vecs[:, 1:k] ** 4, axis=0)))

    # Maximum row norm concentration in k-dimensional spectral embedding (before row-norm)
    ord_row_norms = np.linalg.norm(ord_vecs[:, :k], axis=1)
    sym_row_norms = np.linalg.norm(sym_vecs[:, :k], axis=1)

    min_deg_idx = int(np.argmin(graph.degrees))
    max_deg_idx = int(np.argmax(graph.degrees))

    return {
        "ord_ipr": ord_ipr,
        "sym_ipr": sym_ipr,
        "ord_min_deg_energy": float(ord_row_norms[min_deg_idx] ** 2),
        "ord_max_deg_energy": float(ord_row_norms[max_deg_idx] ** 2),
        "sym_min_deg_energy": float(sym_row_norms[min_deg_idx] ** 2),
        "sym_max_deg_energy": float(sym_row_norms[max_deg_idx] ** 2),
        "ord_gap": float(ord_vals[k] - ord_vals[k - 1]),
        "sym_gap": float(sym_vals[k] - sym_vals[k - 1]),
        "ord_lambda3_over_dmin": float(ord_vals[k - 1] / max(np.min(graph.degrees), 1.0)),
    }


def run_phase_grid(n_trials=30, k=3, kmeans_seed=42):
    p_in = 0.65
    p_out_grid = [0.02, 0.06, 0.10, 0.15, 0.22]
    gamma_grid = [
        ("Balanced", 0.00),
        ("Moderate", 0.45),
        ("High", 0.85),
    ]

    grid_results = []
    for p_out in p_out_grid:
        for cond_name, gamma in gamma_grid:
            prob_matrix, true_labels, theta, cal_info = build_dcsbm_probabilities(
                n_communities=k,
                community_size=10,
                p_in=p_in,
                p_out=p_out,
                gamma=gamma,
                calibrate=True,
            )
            trial_rows = []
            for trial in range(n_trials):
                seed = 2000 + trial
                A = sample_graph_from_probabilities(prob_matrix, seed=seed)
                graph = SpectralGraph(A)
                diag = compute_graph_diagnostics(graph, true_labels, k=k)
                loc = eigenvector_localization_metrics(graph, k=k)

                ord_labels, _, _, _ = ordinary_spectral_clustering(
                    graph, k=k, row_normalize=False, seed=kmeans_seed
                )
                norm_labels, _, _, _, _ = normalized_spectral_clustering_variants(
                    graph, k=k, seed=kmeans_seed
                )
                reg_labels, _ = regularized_normalized_clustering(
                    graph, k=k, seed=kmeans_seed
                )

                ari_ord = float(adjusted_rand_score(true_labels, ord_labels))
                ari_norm = float(adjusted_rand_score(true_labels, norm_labels))
                ari_reg = float(adjusted_rand_score(true_labels, reg_labels))

                trial_rows.append({
                    "ari_ord": ari_ord,
                    "ari_norm": ari_norm,
                    "ari_reg": ari_reg,
                    "delta_ari": ari_norm - ari_ord,
                    **diag,
                    **loc,
                })

            st = paired_statistics([r["delta_ari"] for r in trial_rows])
            grid_results.append({
                "p_in": p_in,
                "p_out": p_out,
                "cond_name": cond_name,
                "gamma": gamma,
                "trials": trial_rows,
                "stats": st,
            })
    return grid_results


def run_sparse_vulnerability_study(n_trials=30, k=3, kmeans_seed=42):
    # Investigate when standard normalized Laplacian loses advantage due to low-degree / pendant nodes
    sparse_configs = [
        ("Dense Assortative (p_in=0.65, p_out=0.08)", 0.65, 0.08, 0.85),
        ("Moderate Sparse (p_in=0.45, p_out=0.05)", 0.45, 0.05, 0.85),
        ("Extreme Sparse Hub-Periphery (p_in=0.32, p_out=0.03)", 0.32, 0.03, 1.00),
    ]
    rows = []
    for label, p_in, p_out, gamma in sparse_configs:
        prob_matrix, true_labels, _, _ = build_dcsbm_probabilities(
            n_communities=k,
            community_size=10,
            p_in=p_in,
            p_out=p_out,
            gamma=gamma,
            calibrate=True,
        )
        recs = []
        for trial in range(n_trials):
            A = sample_graph_from_probabilities(prob_matrix, seed=3000 + trial)
            graph = SpectralGraph(A)
            diag = compute_graph_diagnostics(graph, true_labels, k=k)
            ord_labels, _, _, _ = ordinary_spectral_clustering(graph, k=k, row_normalize=False, seed=kmeans_seed)
            ord_rn_labels, _, _, _ = ordinary_spectral_clustering(graph, k=k, row_normalize=True, seed=kmeans_seed)
            norm_labels, _, _, _, _ = normalized_spectral_clustering_variants(graph, k=k, seed=kmeans_seed)
            reg_labels, _ = regularized_normalized_clustering(graph, k=k, seed=kmeans_seed)
            recs.append({
                "ari_ord": float(adjusted_rand_score(true_labels, ord_labels)),
                "ari_ord_rn": float(adjusted_rand_score(true_labels, ord_rn_labels)),
                "ari_norm": float(adjusted_rand_score(true_labels, norm_labels)),
                "ari_reg": float(adjusted_rand_score(true_labels, reg_labels)),
                **diag,
            })
        rows.append((label, recs))
    return rows


def main():
    print("=" * 94)
    print("DAY 8 RESEARCH INVESTIGATION: REGIME INTERACTION, EIGENVECTOR LOCALIZATION & FAILURE MODES")
    print("=" * 94)

    grid_results = run_phase_grid(n_trials=30, k=3)

    print("\n1. TWO-WAY INTERACTION: DEGREE HETEROGENEITY (GAMMA) x COMMUNITY MIXING (P_OUT)")
    print("-" * 94)
    print(
        f"{'p_out':>5} {'Condition':<10} {'Mix mu':>8} {'Ord ARI':>9} {'Norm ARI':>9} "
        f"{'Delta ARI':>10} {'95% CI':>19} {'p (Wilcox)':>11} {'W/T/L':>8}"
    )
    for item in grid_results:
        trs = item["trials"]
        st = item["stats"]
        mu = np.mean([r["mixing_mu"] for r in trs])
        ord_m = np.mean([r["ari_ord"] for r in trs])
        norm_m = np.mean([r["ari_norm"] for r in trs])
        ci_s = f"[{st['ci_low']:+.3f}, {st['ci_high']:+.3f}]"
        wtl_s = f"{st['wins']}/{st['ties']}/{st['losses']}"
        print(
            f"{item['p_out']:5.2f} {item['cond_name']:<10} {mu:8.4f} {ord_m:9.4f} {norm_m:9.4f} "
            f"{st['mean']:+10.4f} {ci_s:>19} {st['w_pval']:11.3e} {wtl_s:>8}"
        )

    print("\n2. MECHANISTIC SPECTRAL DIAGNOSTICS: EIGENGAP, THRESHOLD RATIO (LAMBDA_3/D_MIN) & IPR")
    print("-" * 94)
    print(
        f"{'p_out':>5} {'Condition':<10} {'Gap(L)':>8} {'Gap(L_sym)':>11} "
        f"{'lam3/d_min':>11} {'IPR(L)':>9} {'IPR(L_sym)':>11} {'MinDeg E(L)':>12}"
    )
    for item in grid_results:
        trs = item["trials"]
        print(
            f"{item['p_out']:5.2f} {item['cond_name']:<10} "
            f"{np.mean([r['ord_gap'] for r in trs]):8.4f} "
            f"{np.mean([r['sym_gap'] for r in trs]):11.4f} "
            f"{np.mean([r['ord_lambda3_over_dmin'] for r in trs]):11.4f} "
            f"{np.mean([r['ord_ipr'] for r in trs]):9.4f} "
            f"{np.mean([r['sym_ipr'] for r in trs]):11.4f} "
            f"{np.mean([r['ord_min_deg_energy'] for r in trs]):12.4f}"
        )

    all_trials = [r for item in grid_results for r in item["trials"]]
    ord_gaps = np.array([r["ord_gap"] for r in all_trials])
    sym_gaps = np.array([r["sym_gap"] for r in all_trials])
    lam_ratios = np.array([r["ord_lambda3_over_dmin"] for r in all_trials])
    ord_iprs = np.array([r["ord_ipr"] for r in all_trials])
    ord_aris = np.array([r["ari_ord"] for r in all_trials])
    norm_aris = np.array([r["ari_norm"] for r in all_trials])
    deltas = np.array([r["delta_ari"] for r in all_trials])

    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    print("\n3. GLOBAL CORRELATIONS ACROSS ALL 450 GRAPH INSTANCES (15 REGIMES x 30 TRIALS)")
    print("-" * 94)
    print(f"Correlation(Gap(L), Ord ARI)              = {corr(ord_gaps, ord_aris):+.4f}")
    print(f"Correlation(Gap(L_sym), Norm ARI)         = {corr(sym_gaps, norm_aris):+.4f}")
    print(f"Correlation(lambda_3(L)/d_min, Ord ARI)   = {corr(lam_ratios, ord_aris):+.4f}")
    print(f"Correlation(IPR(L), Ord ARI)              = {corr(ord_iprs, ord_aris):+.4f}")
    print(f"Correlation(IPR(L), Delta ARI)            = {corr(ord_iprs, deltas):+.4f}")

    print("\n4. SPARSE PERIPHERAL VULNERABILITY & DEGREE-REGULARIZED LAPLACIAN (L_TAU)")
    print("-" * 94)
    sparse_rows = run_sparse_vulnerability_study(n_trials=30, k=3)
    print(
        f"{'Regime Description':<44} {'d_min':>6} {'Comp':>6} "
        f"{'Ord ARI':>9} {'Ord+RN':>8} {'Norm ARI':>9} {'Reg L_tau':>10}"
    )
    for label, recs in sparse_rows:
        print(
            f"{label:<44} "
            f"{np.mean([r['deg_min'] for r in recs]):6.2f} "
            f"{np.mean([r['components'] for r in recs]):6.2f} "
            f"{np.mean([r['ari_ord'] for r in recs]):9.4f} "
            f"{np.mean([r['ari_ord_rn'] for r in recs]):8.4f} "
            f"{np.mean([r['ari_norm'] for r in recs]):9.4f} "
            f"{np.mean([r['ari_reg'] for r in recs]):10.4f}"
        )
    print("=" * 94)


if __name__ == "__main__":
    main()
