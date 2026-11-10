import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from sklearn.metrics import adjusted_rand_score
except ImportError:
    def adjusted_rand_score(labels_true, labels_pred):
        labels_true = np.asarray(labels_true, dtype=int)
        labels_pred = np.asarray(labels_pred, dtype=int)
        n = len(labels_true)
        _, true_idx = np.unique(labels_true, return_inverse=True)
        _, pred_idx = np.unique(labels_pred, return_inverse=True)
        n_true = int(true_idx.max()) + 1
        n_pred = int(pred_idx.max()) + 1
        contingency = np.zeros((n_true, n_pred), dtype=np.int64)
        np.add.at(contingency, (true_idx, pred_idx), 1)
        sum_comb_c = np.sum(contingency * (contingency - 1)) / 2.0
        a = contingency.sum(axis=1)
        b = contingency.sum(axis=0)
        sum_comb_a = np.sum(a * (a - 1)) / 2.0
        sum_comb_b = np.sum(b * (b - 1)) / 2.0
        comb_n = n * (n - 1) / 2.0
        expected = (sum_comb_a * sum_comb_b) / comb_n if comb_n > 0 else 0.0
        max_index = 0.5 * (sum_comb_a + sum_comb_b)
        denom = max_index - expected
        if abs(denom) < 1e-15:
            return 1.0 if abs(sum_comb_c - expected) < 1e-15 else 0.0
        return float((sum_comb_c - expected) / denom)

from src.spectral_core import SpectralGraph


def build_community_propensities(n_communities, community_size, gamma):
    ranks = np.arange(1, community_size + 1, dtype=float)
    profile = ranks ** (-float(gamma))
    profile = profile * (community_size / np.sum(profile))
    return np.tile(profile, n_communities)


def calibrate_scale(products, target_sum, tol=1e-10, max_iter=80):
    if target_sum <= 0:
        return 0.0
    max_possible = float(len(products))
    if target_sum >= max_possible:
        return 1e6

    low = 0.0
    high = max(1.0, target_sum / max(np.sum(products), 1e-12))
    while np.sum(np.minimum(1.0, high * products)) < target_sum and high < 1e6:
        high *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        current = float(np.sum(np.minimum(1.0, mid * products)))
        if abs(current - target_sum) <= tol:
            return mid
        if current < target_sum:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def build_dcsbm_probabilities(n_communities, community_size, p_in, p_out, gamma, calibrate=True):
    n = n_communities * community_size
    labels = np.repeat(np.arange(n_communities), community_size)
    theta = build_community_propensities(n_communities, community_size, gamma)

    in_pairs = []
    out_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                in_pairs.append((i, j, theta[i] * theta[j]))
            else:
                out_pairs.append((i, j, theta[i] * theta[j]))

    in_products = np.array([p[2] for p in in_pairs], dtype=float)
    out_products = np.array([p[2] for p in out_pairs], dtype=float)

    target_m_in = len(in_products) * p_in
    target_m_out = len(out_products) * p_out

    if calibrate:
        scale_in = calibrate_scale(in_products, target_m_in)
        scale_out = calibrate_scale(out_products, target_m_out)
    else:
        scale_in = p_in
        scale_out = p_out

    prob_matrix = np.zeros((n, n), dtype=float)
    saturated_pairs = 0
    for i, j, prod in in_pairs:
        raw_p = scale_in * prod
        if raw_p >= 1.0 - 1e-12:
            saturated_pairs += 1
        p_ij = min(1.0, raw_p)
        prob_matrix[i, j] = p_ij
        prob_matrix[j, i] = p_ij

    for i, j, prod in out_pairs:
        raw_p = scale_out * prod
        if raw_p >= 1.0 - 1e-12:
            saturated_pairs += 1
        p_ij = min(1.0, raw_p)
        prob_matrix[i, j] = p_ij
        prob_matrix[j, i] = p_ij

    calibration_info = {
        "gamma": gamma,
        "theta_cv": float(np.std(theta) / np.mean(theta)),
        "theta_min": float(np.min(theta)),
        "theta_max": float(np.max(theta)),
        "scale_in": float(scale_in),
        "scale_out": float(scale_out),
        "expected_m_in": float(np.sum(np.minimum(1.0, scale_in * in_products))),
        "expected_m_out": float(np.sum(np.minimum(1.0, scale_out * out_products))),
        "saturated_pairs": int(saturated_pairs),
    }
    return prob_matrix, labels, theta, calibration_info


def sample_graph_from_probabilities(prob_matrix, seed):
    rng = np.random.default_rng(seed)
    n = prob_matrix.shape[0]
    adjacency = np.zeros((n, n), dtype=float)
    upper_i, upper_j = np.triu_indices(n, k=1)
    draws = rng.random(len(upper_i))
    edges = draws < prob_matrix[upper_i, upper_j]
    adjacency[upper_i[edges], upper_j[edges]] = 1.0
    adjacency[upper_j[edges], upper_i[edges]] = 1.0
    return adjacency


def ordinary_spectral_clustering(graph, k, row_normalize=False, max_iter=100, seed=42):
    values, vectors = graph.eigenpairs()
    embedding = vectors[:, :k].copy()
    if row_normalize:
        row_norms = np.linalg.norm(embedding, axis=1)
        nonzero = row_norms > 1e-12
        embedding[nonzero] /= row_norms[nonzero, None]
    labels, centers = graph._kmeans(embedding, k, max_iter=max_iter, seed=seed)
    return labels, values, embedding, centers


def normalized_spectral_clustering_variants(graph, k, max_iter=100, seed=42):
    labels_norm, _, embedding_norm, centers_norm = graph.spectral_clustering(
        k=k, max_iter=max_iter, seed=seed
    )
    sym_values, sym_vectors = graph.normalized_eigenpairs()
    embedding_unnorm = sym_vectors[:, :k].copy()
    labels_unnorm, _ = graph._kmeans(
        embedding_unnorm, k, max_iter=max_iter, seed=seed
    )
    return labels_norm, labels_unnorm, sym_values, embedding_norm, centers_norm


def compute_graph_diagnostics(graph, true_labels, k=3):
    A = graph.A
    degrees = graph.degrees
    n = graph.n

    upper_i, upper_j = np.triu_indices(n, k=1)
    same_comm = true_labels[upper_i] == true_labels[upper_j]
    edge_weights = A[upper_i, upper_j]

    m_in = float(np.sum(edge_weights[same_comm]))
    m_out = float(np.sum(edge_weights[~same_comm]))
    total_edges = m_in + m_out
    mixing_mu = m_out / total_edges if total_edges > 0 else 1.0

    conductances = []
    for comm in range(k):
        group = np.where(true_labels == comm)[0]
        rest = np.where(true_labels != comm)[0]
        vol_g = graph.volume(group)
        vol_r = graph.volume(rest)
        denom = min(vol_g, vol_r)
        if denom > 0:
            conductances.append(graph.cut_value(group, rest) / denom)
    mean_conductance = float(np.mean(conductances)) if conductances else 1.0

    deg_mean = float(np.mean(degrees))
    deg_std = float(np.std(degrees))
    deg_cv = deg_std / deg_mean if deg_mean > 0 else 0.0

    return {
        "deg_mean": deg_mean,
        "deg_std": deg_std,
        "deg_cv": deg_cv,
        "deg_min": int(np.min(degrees)),
        "deg_max": int(np.max(degrees)),
        "m_in": int(round(m_in)),
        "m_out": int(round(m_out)),
        "mixing_mu": float(mixing_mu),
        "conductance": mean_conductance,
        "components": int(graph.number_of_components()),
        "isolated": int(np.sum(degrees == 0)),
    }


def paired_statistics(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    median_d = float(np.median(diffs))
    se_d = std_d / math.sqrt(n) if n > 1 else 0.0

    t_crit = 2.04523 if n == 30 else 1.95996
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d

    if se_d > 1e-15:
        t_stat = mean_d / se_d
        try:
            from scipy import stats
            t_pval = float(stats.ttest_1samp(diffs, 0.0).pvalue)
        except Exception:
            t_pval = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    else:
        t_stat = 0.0
        t_pval = 1.0

    nonzero = diffs[np.abs(diffs) > 1e-12]
    if len(nonzero) > 0:
        try:
            from scipy import stats
            res = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
            w_stat = float(res.statistic)
            w_pval = float(res.pvalue)
        except Exception:
            abs_nz = np.abs(nonzero)
            order = np.argsort(abs_nz)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(nonzero) + 1, dtype=float)
            w_pos = float(np.sum(ranks[nonzero > 0]))
            w_neg = float(np.sum(ranks[nonzero < 0]))
            w_stat = min(w_pos, w_neg)
            m = len(nonzero)
            mean_w = m * (m + 1) / 4.0
            std_w = math.sqrt(m * (m + 1) * (2 * m + 1) / 24.0)
            z = (w_stat - mean_w) / std_w if std_w > 0 else 0.0
            w_pval = float(math.erfc(abs(z) / math.sqrt(2.0)))
    else:
        w_stat = 0.0
        w_pval = 1.0

    wins = int(np.sum(diffs > 1e-12))
    ties = int(np.sum(np.abs(diffs) <= 1e-12))
    losses = int(np.sum(diffs < -1e-12))

    return {
        "mean": mean_d,
        "std": std_d,
        "median": median_d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "t_pval": t_pval,
        "w_stat": w_stat,
        "w_pval": w_pval,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def evaluate_condition(name, gamma, p_in, p_out, n_trials=30, k=3, kmeans_seed=42):
    prob_matrix, true_labels, theta, cal_info = build_dcsbm_probabilities(
        n_communities=k,
        community_size=10,
        p_in=p_in,
        p_out=p_out,
        gamma=gamma,
        calibrate=True,
    )

    records = []
    for trial in range(n_trials):
        graph_seed = 1000 + trial
        adjacency = sample_graph_from_probabilities(prob_matrix, seed=graph_seed)
        graph = SpectralGraph(adjacency)

        diag = compute_graph_diagnostics(graph, true_labels, k=k)

        ord_labels, ord_vals, _, _ = ordinary_spectral_clustering(
            graph, k=k, row_normalize=False, seed=kmeans_seed
        )
        ord_rn_labels, _, _, _ = ordinary_spectral_clustering(
            graph, k=k, row_normalize=True, seed=kmeans_seed
        )
        norm_labels, norm_unnorm_labels, sym_vals, _, _ = (
            normalized_spectral_clustering_variants(
                graph, k=k, seed=kmeans_seed
            )
        )

        ari_ord = float(adjusted_rand_score(true_labels, ord_labels))
        ari_ord_rn = float(adjusted_rand_score(true_labels, ord_rn_labels))
        ari_norm = float(adjusted_rand_score(true_labels, norm_labels))
        ari_norm_unnorm = float(adjusted_rand_score(true_labels, norm_unnorm_labels))

        # Zero-based indexing: lambda_4 - lambda_3 is values[3] - values[2]
        eigengap_ord = float(ord_vals[3] - ord_vals[2])
        eigengap_norm = float(sym_vals[3] - sym_vals[2])

        record = {
            "trial": trial,
            "seed": graph_seed,
            "ari_ord": ari_ord,
            "ari_ord_rn": ari_ord_rn,
            "ari_norm": ari_norm,
            "ari_norm_unnorm": ari_norm_unnorm,
            "delta_ari": ari_norm - ari_ord,
            "eigengap_ord": eigengap_ord,
            "eigengap_norm": eigengap_norm,
            **diag,
        }
        records.append(record)

    return {
        "name": name,
        "gamma": gamma,
        "calibration": cal_info,
        "records": records,
    }


def main():
    p_in = 0.65
    p_out = 0.08
    n_trials = 30
    k = 3

    conditions = [
        ("Balanced", 0.00),
        ("Moderately Imbalanced", 0.45),
        ("Highly Imbalanced", 0.85),
    ]

    print("=" * 88)
    print("DAY 7 EXPERIMENT: DEGREE HETEROGENEITY IN SPECTRAL GRAPH CLUSTERING")
    print("=" * 88)
    print(
        f"Graph Model: Separation-Calibrated Degree-Corrected SBM "
        f"(N=30, K=3, n_c=10, trials={n_trials})"
    )
    print(
        f"Baseline Parameters: p_in={p_in:.2f}, p_out={p_out:.2f} "
        f"(Target E[m_in]={3 * 45 * p_in:.2f}, Target E[m_out]={3 * 100 * p_out:.2f})"
    )
    print("-" * 88)

    results = [
        evaluate_condition(name, gamma, p_in=p_in, p_out=p_out, n_trials=n_trials, k=k)
        for name, gamma in conditions
    ]

    print("\n1. PROPENSITY CALIBRATION & EXPECTED EDGE DENSITY CONTROLS")
    print("-" * 88)
    print(
        f"{'Condition':<23} {'gamma':>6} {'Theta CV':>9} {'Theta[min,max]':>16} "
        f"{'E[m_in]':>9} {'E[m_out]':>9} {'Sat.Pairs':>10}"
    )
    for res in results:
        cal = res["calibration"]
        t_range = f"[{cal['theta_min']:.2f}, {cal['theta_max']:.2f}]"
        print(
            f"{res['name']:<23} {cal['gamma']:6.2f} {cal['theta_cv']:9.4f} {t_range:>16} "
            f"{cal['expected_m_in']:9.2f} {cal['expected_m_out']:9.2f} {cal['saturated_pairs']:10d}"
        )

    print("\n2. REALIZED GRAPH DEGREE & COMMUNITY SEPARATION DIAGNOSTICS (MEAN +/- SD)")
    print("-" * 88)
    print(
        f"{'Condition':<23} {'Deg Mean':>9} {'Deg SD':>9} {'Deg CV':>9} "
        f"{'Min/Max':>10} {'m_in':>8} {'m_out':>8} {'Mix mu':>9} {'Comp/Iso':>10}"
    )
    for res in results:
        recs = res["records"]
        d_mean = np.mean([r["deg_mean"] for r in recs])
        d_std = np.mean([r["deg_std"] for r in recs])
        d_cv = np.mean([r["deg_cv"] for r in recs])
        d_cv_sd = np.std([r["deg_cv"] for r in recs])
        d_min = np.mean([r["deg_min"] for r in recs])
        d_max = np.mean([r["deg_max"] for r in recs])
        m_in = np.mean([r["m_in"] for r in recs])
        m_out = np.mean([r["m_out"] for r in recs])
        mu = np.mean([r["mixing_mu"] for r in recs])
        n_disc = sum(1 for r in recs if r["components"] > 1)
        n_iso = sum(1 for r in recs if r["isolated"] > 0)
        minmax_str = f"{d_min:4.1f}/{d_max:4.1f}"
        comp_str = f"{n_disc:2d}d/{n_iso:2d}i"
        print(
            f"{res['name']:<23} {d_mean:9.2f} {d_std:9.2f} "
            f"{d_cv:5.3f}+/-{d_cv_sd:.2f} {minmax_str:>10} "
            f"{m_in:8.2f} {m_out:8.2f} {mu:9.4f} {comp_str:>10}"
        )

    print("\n3. CLUSTERING ACCURACY COMPARISON (ADJUSTED RAND INDEX, ALL 30 TRIALS)")
    print("-" * 88)
    print(
        f"{'Condition':<23} {'Ord ARI (Mean+/-SD)':>20} {'Ord Med':>8} "
        f"{'Norm ARI (Mean+/-SD)':>21} {'Norm Med':>9} {'Delta ARI (Mean+/-SD)':>22}"
    )
    for res in results:
        recs = res["records"]
        ord_aris = np.array([r["ari_ord"] for r in recs])
        norm_aris = np.array([r["ari_norm"] for r in recs])
        deltas = np.array([r["delta_ari"] for r in recs])
        print(
            f"{res['name']:<23} "
            f"{np.mean(ord_aris):9.4f} +/- {np.std(ord_aris, ddof=1):.4f} "
            f"{np.median(ord_aris):8.4f} "
            f"{np.mean(norm_aris):10.4f} +/- {np.std(norm_aris, ddof=1):.4f} "
            f"{np.median(norm_aris):9.4f} "
            f"{np.mean(deltas):+10.4f} +/- {np.std(deltas, ddof=1):.4f}"
        )

    print("\n4. PAIRED STATISTICAL ANALYSIS (DELTA ARI = ARI_NORM - ARI_ORD)")
    print("-" * 88)
    print(
        f"{'Condition':<23} {'95% CI for Delta':>21} {'Paired t':>10} {'p (t-test)':>12} "
        f"{'Wilcoxon W':>11} {'p (Wilcox)':>12} {'W/T/L':>10}"
    )
    for res in results:
        deltas = [r["delta_ari"] for r in res["records"]]
        st = paired_statistics(deltas)
        ci_str = f"[{st['ci_low']:+.4f}, {st['ci_high']:+.4f}]"
        wtl_str = f"{st['wins']}/{st['ties']}/{st['losses']}"
        print(
            f"{res['name']:<23} {ci_str:>21} {st['t_stat']:10.4f} {st['t_pval']:12.4e} "
            f"{st['w_stat']:11.1f} {st['w_pval']:12.4e} {wtl_str:>10}"
        )

    print("\n5. CONNECTED-GRAPH SUBSET VERIFICATION & FAILURE-MODE OVERLAP")
    print("-" * 88)
    print(
        f"{'Condition':<23} {'Conn.Trials':>11} {'Conn.Delta ARI':>15} "
        f"{'Ord Fails(<0.5)':>16} {'Norm Fails(<0.5)':>17} {'Joint Fails':>12}"
    )
    for res in results:
        recs = res["records"]
        conn = [r for r in recs if r["components"] == 1 and r["isolated"] == 0]
        conn_delta = np.mean([r["delta_ari"] for r in conn]) if conn else float("nan")
        ord_fail = sum(1 for r in recs if r["ari_ord"] < 0.5)
        norm_fail = sum(1 for r in recs if r["ari_norm"] < 0.5)
        joint_fail = sum(1 for r in recs if r["ari_ord"] < 0.5 and r["ari_norm"] < 0.5)
        print(
            f"{res['name']:<23} {len(conn):11d} {conn_delta:+15.4f} "
            f"{ord_fail:16d} {norm_fail:17d} {joint_fail:12d}"
        )

    print("\n6. EIGENGAP (LAMBDA_4 - LAMBDA_3) & ROW-NORMALIZATION ABLATION")
    print("-" * 88)
    print(
        f"{'Condition':<23} {'Gap L (Mean+/-SD)':>18} {'Gap L_sym (Mean+/-SD)':>22} "
        f"{'Ord+RowNorm ARI':>16} {'Norm-NoRowNorm ARI':>19}"
    )
    for res in results:
        recs = res["records"]
        g_ord = [r["eigengap_ord"] for r in recs]
        g_norm = [r["eigengap_norm"] for r in recs]
        ari_ord_rn = np.mean([r["ari_ord_rn"] for r in recs])
        ari_norm_un = np.mean([r["ari_norm_unnorm"] for r in recs])
        print(
            f"{res['name']:<23} "
            f"{np.mean(g_ord):8.4f}+/-{np.std(g_ord, ddof=1):.4f} "
            f"{np.mean(g_norm):12.4f}+/-{np.std(g_norm, ddof=1):.4f} "
            f"{ari_ord_rn:16.4f} {ari_norm_un:19.4f}"
        )

    print("\n7. STRATIFIED ANALYSIS BY REALIZED MIXING PARAMETER (MU_EMP = M_OUT / M_TOTAL)")
    print("-" * 88)
    all_recs = [r for res in results for r in res["records"]]
    mus = np.array([r["mixing_mu"] for r in all_recs])
    q1, q2 = np.quantile(mus, [1.0 / 3.0, 2.0 / 3.0])
    strata = [
        ("Low Mixing (mu <= Q33)", lambda mu: mu <= q1),
        ("Mid Mixing (Q33 < mu <= Q67)", lambda mu: q1 < mu <= q2),
        ("High Mixing (mu > Q67)", lambda mu: mu > q2),
    ]
    print(
        f"{'Mixing Stratum':<28} {'Condition':<23} {'n':>4} "
        f"{'Mean mu':>9} {'Ord ARI':>9} {'Norm ARI':>9} {'Delta ARI':>10}"
    )
    for s_name, pred in strata:
        for res in results:
            sub = [r for r in res["records"] if pred(r["mixing_mu"])]
            if not sub:
                continue
            print(
                f"{s_name:<28} {res['name']:<23} {len(sub):4d} "
                f"{np.mean([r['mixing_mu'] for r in sub]):9.4f} "
                f"{np.mean([r['ari_ord'] for r in sub]):9.4f} "
                f"{np.mean([r['ari_norm'] for r in sub]):9.4f} "
                f"{np.mean([r['delta_ari'] for r in sub]):+10.4f}"
            )
    print("=" * 88)


if __name__ == "__main__":
    main()
