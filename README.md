# Spectral Graph Clustering

A from-scratch implementation of spectral graph methods using Python and NumPy. This project explores how graph Laplacians and their eigenvectors reveal community structure in undirected, non-negatively weighted graphs.

## Concepts

- Adjacency and degree matrices
- Ordinary graph Laplacian: L = D - A
- Symmetric normalized Laplacian
- Laplacian eigenvalues and eigenvectors
- Fiedler vector and algebraic connectivity
- Quadratic form and its edge-based interpretation
- Connected components and zero eigenvalues
- Graph cuts, volume, and normalized cut
- Threshold-based two-way partitioning
- Spectral embeddings for multi-community clustering
- K-means clustering of embedded vertices
- Clustering evaluation using Adjusted Rand Index (ARI)
- Robustness testing on randomly generated graphs

## Implementation

The `SpectralGraph` class constructs graph matrices, computes eigenpairs, checks Laplacian properties, evaluates quadratic forms, counts connected components, and generates candidate graph partitions.

Both ordinary and normalized Laplacians are supported. The normalized Laplacian construction handles isolated vertices without division-by-zero errors.

For multi-community clustering, eigenvectors of the normalized Laplacian form a low-dimensional representation of the vertices. K-means groups the embedded vertices into the requested number of clusters.

## Experiments

- `bridge_strength.py`: studies how adding edges between two dense communities affects spectral connectivity and normalized cut.
- `compare_laplacians.py`: compares partitions obtained from ordinary and normalized Laplacians.
- `multi_community.py`: tests three-community clustering on a graph containing dense groups connected by weak bridges.
- `clustering_robustness.py`: evaluates clustering across randomly generated graphs with different within-community and between-community edge probabilities.

The robustness experiment uses three planted communities of ten vertices each. It varies the probabilities of edges within and between communities and measures agreement with the known labels using ARI. Multiple random graph instances are evaluated for each setting.

## Project Structure

```text
spectral-graph-clustering/
├── src/
│   ├── __init__.py
│   └── spectral_core.py
├── experiments/
│   ├── __init__.py
│   ├── bridge_strength.py
│   ├── compare_laplacians.py
│   ├── multi_community.py
│   └── clustering_robustness.py
├── results/
│   ├── bridge_strength.txt
│   ├── laplacian_comparison.txt
│   ├── multi_community.txt
│   └── clustering_robustness.txt
├── requirements.txt
└── README.md
```

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the main demonstration:

```bash
python3 -m src.spectral_core
```

Run the experiments from the project root:

```bash
python3 -m experiments.bridge_strength
python3 -m experiments.compare_laplacians
python3 -m experiments.multi_community
python3 -m experiments.clustering_robustness
```

Save robustness experiment output:

```bash
python3 -m experiments.clustering_robustness | tee results/clustering_robustness.txt
```

## Limitations

The implementation focuses on undirected graphs with non-negative edge weights and no self-loops. Spectral thresholding does not guarantee a globally optimal cut. Multi-community clustering depends on the chosen embedding dimensions and k-means initialization.

Synthetic experiments provide controlled tests but do not establish performance on every real-world graph. When within-community and between-community edge probabilities are equal, the planted labels generally cannot be inferred reliably from the graph structure alone.

## Tools

Python, NumPy, scikit-learn, and Git.


## Day 7 & Day 8 Research Investigation: Degree Heterogeneity and Regime Interactions

### Day 7 Controlled Degree-Heterogeneity Benchmark ($N=30, K=3, n_c=10, p_{\mathrm{in}}=0.65, p_{\mathrm{out}}=0.08$, 30 trials/regime)

Using a Separation-Calibrated Degree-Corrected Stochastic Block Model (`experiments/degree_heterogeneity.py`) that holds expected within-community edges ($\mathbb{E}[m_{\mathrm{in}}] = 87.75$) and between-community edges ($\mathbb{E}[m_{\mathrm{out}}] = 24.00$) strictly invariant across degree-imbalance regimes:

| Regime ($\gamma$) | $\mathrm{CV}_\theta$ | Realized $\bar{d} \pm s_d$ | Realized $\mathrm{CV}_d$ | $\bar{m}_{\mathrm{in}} / \bar{m}_{\mathrm{out}}$ | Mix $\bar{\mu}_{\mathrm{emp}}$ | Ord ARI (Mean $\pm$ SD) | Norm ARI (Mean $\pm$ SD) | Paired $\Delta\mathrm{ARI}$ | 95% CI | Paired $p$ ($t$ / Wilcox) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Balanced ($0.00$) | $0.0000$ | $7.41 \pm 1.79$ | $0.244 \pm 0.05$ | $87.13 / 24.03$ | $0.2164$ | $0.9680 \pm 0.0925$ | $0.9900 \pm 0.0401$ | $+0.0220 \pm 0.0860$ | $[-0.0101, +0.0541]$ | $0.1718$ / $0.2500$ |
| Moderate ($0.45$) | $0.3572$ | $7.40 \pm 2.50$ | $0.340 \pm 0.05$ | $87.53 / 23.53$ | $0.2113$ | $0.9170 \pm 0.1768$ | $0.9864 \pm 0.0352$ | $+0.0694 \pm 0.1673$ | $[+0.0069, +0.1318]$ | $0.0307$ / $0.0313$ |
| High ($0.85$) | $0.7427$ | $7.39 \pm 3.11$ | $0.423 \pm 0.04$ | $87.40 / 23.47$ | $0.2112$ | $0.9073 \pm 0.1812$ | $1.0000 \pm 0.0000$ | $+0.0927 \pm 0.1812$ | $[+0.0251, +0.1604]$ | $0.0089$ / $0.0156$ |

### Day 8 Two-Way Regime Interaction, Eigenvector Localization, & Sparse Regularization (`experiments/regime_interaction_study.py`)

- **Strong Separation ($p_{\mathrm{out}}=0.02, \bar{\mu}\approx 0.06$):** Both $L$ and $L_{\mathrm{sym}}$ achieve $\mathrm{ARI} = 1.0000$ ($\Delta\mathrm{ARI} = 0.0000$) even under high degree heterogeneity because $\lambda_3(L)/d_{\min} \le 0.2523 \ll 1$.
- **Elevated Mixing ($p_{\mathrm{out}}=0.22, \bar{\mu}\approx 0.42$):** As $\lambda_3(L)/d_{\min} \to 0.9977$ under High degree imbalance, $\mathrm{IPR}(L)$ surges to $0.3316$ (with the lowest-degree vertex absorbing $71.04\%$ of an eigenvector's squared norm, correlating with $\mathrm{ARI}_{\mathrm{ord}}$ at $r = -0.7494$ across all 450 graphs), causing $\mathrm{ARI}_{\mathrm{ord}}$ to drop to $0.3335$ while $\mathrm{ARI}_{\mathrm{norm}}$ achieves $0.8844$ ($\Delta\mathrm{ARI} = +0.5509$, $95\%$ CI $[+0.462, +0.640]$, Wilcoxon $p = 1.73 \times 10^{-6}$).
- **Sparse Breakdown of $L_{\mathrm{sym}}$ & Regularized Recovery:** In the extreme sparse hub-periphery regime ($p_{\mathrm{in}}=0.32, p_{\mathrm{out}}=0.03, \gamma=1.00, \bar{d}_{\min}=0.17$), standard $L_{\mathrm{sym}}$ drops to $\mathrm{ARI} = 0.2134$ due to disconnected and pendant vertices, whereas degree-regularized $L_\tau$ ($\tau = 0.25\bar{d}$) restores accuracy to $\mathrm{ARI} = 0.7357$.

### Reproduction & Research Paper

```bash
python3 -m unittest discover -s tests -v
python3 -m experiments.degree_heterogeneity | tee results/degree_heterogeneity.txt
python3 -m experiments.regime_interaction_study | tee results/regime_interaction_study.txt
python3 -m experiments.generate_paper_figures
```

- **Compiled Research Paper PDF:** [`spectral_graph_clustering_research_paper.pdf`](spectral_graph_clustering_research_paper.pdf)
- **LaTeX Source & Bibliography:** [`paper/main.tex`](paper/main.tex) and [`paper/references.bib`](paper/references.bib)
