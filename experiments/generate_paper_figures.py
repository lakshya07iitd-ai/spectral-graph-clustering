from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.degree_heterogeneity import evaluate_condition
from experiments.regime_interaction_study import (
    run_phase_grid,
    run_sparse_vulnerability_study,
)


def set_pub_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def main():
    set_pub_style()
    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    day7_conditions = [
        ("Balanced\n(gamma=0.00)", 0.00),
        ("Moderate\n(gamma=0.45)", 0.45),
        ("High\n(gamma=0.85)", 0.85),
    ]
    day7_res = [
        evaluate_condition(name, gamma, p_in=0.65, p_out=0.08, n_trials=30, k=3)
        for name, gamma in day7_conditions
    ]

    # Figure 1: Day 7 ARI comparison and Eigengap collapse
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.75))
    x = np.arange(len(day7_res))
    width = 0.25

    ord_means = [np.mean([r["ari_ord"] for r in res["records"]]) for res in day7_res]
    ord_sds = [np.std([r["ari_ord"] for r in res["records"]], ddof=1) / np.sqrt(30) for res in day7_res]
    rn_means = [np.mean([r["ari_ord_rn"] for r in res["records"]]) for res in day7_res]
    rn_sds = [np.std([r["ari_ord_rn"] for r in res["records"]], ddof=1) / np.sqrt(30) for res in day7_res]
    norm_means = [np.mean([r["ari_norm"] for r in res["records"]]) for res in day7_res]
    norm_sds = [np.std([r["ari_norm"] for r in res["records"]], ddof=1) / np.sqrt(30) for res in day7_res]

    ax = axes[0]
    ax.bar(x - width, ord_means, width, yerr=ord_sds, capsize=3, label="Ordinary L", color="#C0392B", alpha=0.88)
    ax.bar(x, rn_means, width, yerr=rn_sds, capsize=3, label="Ordinary L + RowNorm", color="#E67E22", alpha=0.88)
    ax.bar(x + width, norm_means, width, yerr=norm_sds, capsize=3, label="Normalized L_sym", color="#1F618D", alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in day7_conditions])
    ax.set_ylim(0.75, 1.03)
    ax.set_ylabel("Adjusted Rand Index (Mean +/- SE)")
    ax.set_title("(a) Day 7 Accuracy by Degree Regime", pad=8)
    ax.legend(loc="lower left", frameon=True)

    ax2 = axes[1]
    gap_l = [np.mean([r["eigengap_ord"] for r in res["records"]]) for res in day7_res]
    gap_l_se = [np.std([r["eigengap_ord"] for r in res["records"]], ddof=1) / np.sqrt(30) for res in day7_res]
    gap_sym = [np.mean([r["eigengap_norm"] for r in res["records"]]) for res in day7_res]
    gap_sym_se = [np.std([r["eigengap_norm"] for r in res["records"]], ddof=1) / np.sqrt(30) for res in day7_res]

    ax2.errorbar(x, gap_l, yerr=gap_l_se, marker="o", linewidth=2, capsize=4, color="#C0392B", label="Ordinary Gap (L)")
    ax2_twin = ax2.twinx()
    ax2_twin.spines["top"].set_visible(False)
    ax2_twin.grid(False)
    ax2_twin.errorbar(x, gap_sym, yerr=gap_sym_se, marker="s", linestyle="--", linewidth=2, capsize=4, color="#1F618D", label="Normalized Gap (L_sym)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[0] for c in day7_conditions])
    ax2.set_ylabel("Ordinary Eigengap (L)", color="#C0392B")
    ax2_twin.set_ylabel("Normalized Eigengap (L_sym)", color="#1F618D")
    ax2.set_ylim(0.4, 1.7)
    ax2_twin.set_ylim(0.20, 0.45)
    ax2.set_title("(b) Eigengap Compression vs. Stability", pad=8)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=7.5)

    fig.tight_layout(w_pad=2.8)
    fig.savefig(fig_dir / "fig1_day7_ari_eigengap.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig1_day7_ari_eigengap.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Day 8 Phase Grid data
    grid = run_phase_grid(n_trials=30, k=3)
    p_outs = [0.02, 0.06, 0.10, 0.15, 0.22]
    conds = ["Balanced", "Moderate", "High"]
    colors = {"Balanced": "#27AE60", "Moderate": "#E67E22", "High": "#8E44AD"}
    markers = {"Balanced": "o", "Moderate": "s", "High": "^"}

    # Figure 2: Interaction between Degree Heterogeneity and Community Mixing
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.75))
    ax = axes[0]
    for cond in conds:
        sub = [g for g in grid if g["cond_name"] == cond]
        ord_y = [np.mean([r["ari_ord"] for r in g["trials"]]) for g in sub]
        norm_y = [np.mean([r["ari_norm"] for r in g["trials"]]) for g in sub]
        ax.plot(p_outs, ord_y, marker=markers[cond], linestyle="--", linewidth=1.6, color=colors[cond], label=f"Ord ({cond})")
        ax.plot(p_outs, norm_y, marker=markers[cond], linestyle="-", linewidth=1.9, color=colors[cond], label=f"Norm ({cond})")
    ax.set_xlabel("Between-Community Probability (p_out)")
    ax.set_ylabel("Mean Adjusted Rand Index (ARI)")
    ax.set_title("(a) Ordinary vs. Normalized Recovery", pad=8)
    ax.set_ylim(0.25, 1.04)
    ax.legend(ncol=2, fontsize=6.9, loc="lower left", frameon=True)

    ax2 = axes[1]
    for cond in conds:
        sub = [g for g in grid if g["cond_name"] == cond]
        deltas = [g["stats"]["mean"] for g in sub]
        ci_low = [g["stats"]["ci_low"] for g in sub]
        ci_high = [g["stats"]["ci_high"] for g in sub]
        yerr = [np.array(deltas) - np.array(ci_low), np.array(ci_high) - np.array(deltas)]
        ax2.errorbar(p_outs, deltas, yerr=yerr, marker=markers[cond], linewidth=1.9, capsize=3, color=colors[cond], label=f"{cond} Imbalance")
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("Between-Community Probability (p_out)")
    ax2.set_ylabel("Paired Gain Delta ARI (Norm - Ord)")
    ax2.set_title("(b) Two-Way Interaction in Delta ARI", pad=8)
    ax2.legend(loc="upper left", frameon=True)

    fig.tight_layout(w_pad=2.8)
    fig.savefig(fig_dir / "fig2_phase_interaction.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig2_phase_interaction.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Eigenvector Localization Mechanism (IPR and lambda_3 / d_min)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.75))
    ax = axes[0]
    for cond in conds:
        sub = [g for g in grid if g["cond_name"] == cond]
        ratios = [np.mean([r["ord_lambda3_over_dmin"] for r in g["trials"]]) for g in sub]
        iprs = [np.mean([r["ord_ipr"] for r in g["trials"]]) for g in sub]
        ax.plot(ratios, iprs, marker=markers[cond], linewidth=2.0, color=colors[cond], label=f"Ordinary L ({cond})")
    for cond in conds:
        sub = [g for g in grid if g["cond_name"] == cond]
        ratios = [np.mean([r["ord_lambda3_over_dmin"] for r in g["trials"]]) for g in sub]
        sym_iprs = [np.mean([r["sym_ipr"] for r in g["trials"]]) for g in sub]
        ax.plot(ratios, sym_iprs, marker=markers[cond], linestyle=":", linewidth=1.4, color="#1F618D", alpha=0.75 if cond != "High" else 1.0, label="Normalized L_sym" if cond == "High" else None)
    ax.axhline(1.0 / 30.0, color="gray", linestyle="--", linewidth=0.9, label="Uniform Floor (1/N = 0.033)")
    ax.set_xlabel("Spectral Threshold Ratio E[lambda_3(L) / d_min]")
    ax.set_ylabel("Inverse Participation Ratio (IPR)")
    ax.set_title("(a) Eigenvector Localization vs. Threshold", pad=8)
    ax.legend(loc="upper left", fontsize=7.1, frameon=True)

    ax2 = axes[1]
    all_trials = [r for g in grid for r in g["trials"]]
    ipr_vals = np.array([r["ord_ipr"] for r in all_trials])
    delta_vals = np.array([r["delta_ari"] for r in all_trials])
    ratio_vals = np.array([r["ord_lambda3_over_dmin"] for r in all_trials])
    sc = ax2.scatter(ipr_vals, delta_vals, c=ratio_vals, cmap="viridis", s=16, alpha=0.72, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax2)
    cb.set_label("lambda_3(L) / d_min", fontsize=8.0)
    m, b = np.polyfit(ipr_vals, delta_vals, 1)
    xs = np.linspace(ipr_vals.min(), ipr_vals.max(), 100)
    ax2.plot(xs, m * xs + b, color="#C0392B", linewidth=1.6, linestyle="--", label="Linear Fit (r = +0.745)")
    ax2.set_xlabel("Ordinary Laplacian Eigenvector IPR(L)")
    ax2.set_ylabel("Paired Gain Delta ARI (Norm - Ord)")
    ax2.set_title("(b) Localization Drives Delta ARI (n=450)", pad=8)
    ax2.legend(loc="upper left", frameon=True)

    fig.tight_layout(w_pad=2.8)
    fig.savefig(fig_dir / "fig3_localization_mechanism.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig3_localization_mechanism.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: Sparse Peripheral Vulnerability & Degree Regularization
    sparse_rows = run_sparse_vulnerability_study(n_trials=30, k=3)
    fig, ax = plt.subplots(figsize=(6.4, 2.65))
    short_labels = [
        "Dense Assortative\n(p_in=0.65, d_min=2.90)",
        "Moderate Sparse\n(p_in=0.45, d_min=0.97)",
        "Extreme Sparse\n(p_in=0.32, d_min=0.17)",
    ]
    x = np.arange(len(sparse_rows))
    w = 0.20
    m_ord = [np.mean([r["ari_ord"] for r in recs]) for _, recs in sparse_rows]
    m_rn = [np.mean([r["ari_ord_rn"] for r in recs]) for _, recs in sparse_rows]
    m_norm = [np.mean([r["ari_norm"] for r in recs]) for _, recs in sparse_rows]
    m_reg = [np.mean([r["ari_reg"] for r in recs]) for _, recs in sparse_rows]

    se_ord = [np.std([r["ari_ord"] for r in recs], ddof=1) / np.sqrt(30) for _, recs in sparse_rows]
    se_rn = [np.std([r["ari_ord_rn"] for r in recs], ddof=1) / np.sqrt(30) for _, recs in sparse_rows]
    se_norm = [np.std([r["ari_norm"] for r in recs], ddof=1) / np.sqrt(30) for _, recs in sparse_rows]
    se_reg = [np.std([r["ari_reg"] for r in recs], ddof=1) / np.sqrt(30) for _, recs in sparse_rows]

    ax.bar(x - 1.5 * w, m_ord, w, yerr=se_ord, capsize=3, label="Ordinary L", color="#C0392B", alpha=0.88)
    ax.bar(x - 0.5 * w, m_rn, w, yerr=se_rn, capsize=3, label="Ordinary L + RowNorm", color="#E67E22", alpha=0.88)
    ax.bar(x + 0.5 * w, m_norm, w, yerr=se_norm, capsize=3, label="Standard Normalized L_sym", color="#1F618D", alpha=0.90)
    ax.bar(x + 1.5 * w, m_reg, w, yerr=se_reg, capsize=3, label="Regularized L_tau (tau = 0.25 d_bar)", color="#117864", alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Adjusted Rand Index (Mean +/- SE)")
    ax.set_title("Sparse Breakdown of Standard L_sym & Recovery via Regularization", pad=8)
    ax.legend(ncol=2, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_sparse_regularization.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig4_sparse_regularization.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
