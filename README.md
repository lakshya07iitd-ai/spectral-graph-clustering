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
