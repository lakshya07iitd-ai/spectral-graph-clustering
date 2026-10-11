# Spectral Graph Clustering

A from-scratch implementation of spectral graph methods using Python and NumPy. The project explores how graph Laplacians and their eigenvectors can be used to identify communities in undirected graphs.

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
- Comparison of ordinary and normalized spectral methods

## Implementation

The `SpectralGraph` class constructs graph matrices, computes eigenpairs, checks Laplacian properties, evaluates quadratic forms, counts connected components, and generates candidate graph partitions.

Both ordinary and normalized Laplacians are supported. The normalized Laplacian construction handles isolated vertices without division-by-zero errors.

For multi-community clustering, the implementation uses eigenvectors of the normalized Laplacian to create a low-dimensional representation of the vertices. K-means then groups the vertices into the requested number of clusters.

## Experiments

- `bridge_strength.py`: studies how adding edges between two dense communities affects spectral connectivity and normalized cut.
- `compare_laplacians.py`: compares partitions obtained from ordinary and normalized Laplacians on the same graph.
- `multi_community.py`: tests clustering on a graph containing three dense communities connected by weak bridges.

In the multi-community experiment, nine vertices are divided into three dense groups connected by edges of weight 0.2. The graph remains connected, while its small normalized Laplacian eigenvalues reflect its weak community connections. The tested implementation recovered all three groups, each containing three vertices.

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
│   └── multi_community.py
├── results/
│   ├── bridge_strength.txt
│   ├── laplacian_comparison.txt
│   └── multi_community.txt
├── requirements.txt
└── README.md
```

## Setup

Install the dependencies:

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
```

## Limitations

The project focuses on undirected, non-negatively weighted graphs without self-loops. Spectral thresholding searches candidate two-way partitions derived from an eigenvector and does not guarantee a globally optimal cut. Multi-community clustering depends on the selected embedding dimensions and k-means initialization. The normalized method is not guaranteed to outperform the ordinary method on every graph.

## Tools

Python, NumPy, and Git.
