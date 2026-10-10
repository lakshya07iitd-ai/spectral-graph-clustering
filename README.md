
# Spectral Graph Clustering

A from-scratch implementation of spectral graph methods using Python and NumPy. The project explores how graph Laplacians and their eigenvectors can be used to identify communities in an undirected graph.

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
- Comparison of ordinary and normalized spectral methods

## Implementation

The `SpectralGraph` class constructs the graph matrices, computes eigenpairs, checks Laplacian properties, evaluates quadratic forms, counts connected components, and generates candidate two-way partitions.

Both ordinary and normalized Laplacians are supported. Normalized Laplacian construction handles isolated vertices without division-by-zero errors.

## Experiments

- `bridge_strength.py`: studies how adding edges between two dense communities affects spectral connectivity and normalized cut.
- `compare_laplacians.py`: compares partitions obtained from ordinary and normalized Laplacians on the same graph.

## Project Structure

```text
spectral-graph-clustering/
├── src/
│   ├── __init__.py
│   └── spectral_core.py
├── experiments/
│   ├── __init__.py
│   ├── bridge_strength.py
│   └── compare_laplacians.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Run the main demonstration:

```bash
python3 -m src.spectral_core
```

Run the experiments:

```bash
python3 experiments/bridge_strength.py
python3 experiments/compare_laplacians.py
```

## Limitations

The project focuses on two-way partitioning of undirected, non-negatively weighted graphs without self-loops. Spectral thresholding searches only candidate partitions derived from an eigenvector; it does not guarantee a globally optimal cut. The normalized method is not guaranteed to outperform the ordinary method on every graph.

## Tools

Python, NumPy, and Git.
