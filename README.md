# Spectral Graph Clustering

A from-scratch implementation of spectral graph methods using Python and NumPy. The project studies how the eigenstructure of a graph Laplacian can be used to identify communities in a graph.

## Concepts

* Adjacency matrix and degree matrix
* Graph Laplacian: \(L = D-A\)
* Laplacian eigenvalues and eigenvectors
* Fiedler vector and algebraic connectivity
* Laplacian quadratic form and its edge-based interpretation
* Connected components and zero eigenvalues
* Graph cuts and two-way graph partitioning
* Normalized cut and volume-based partition evaluation
* Threshold selection using the Fiedler vector

## Implementation

The `SpectralGraph` class provides methods for constructing and checking graph matrices, computing eigenpairs, extracting the Fiedler vector, evaluating quadratic forms, counting connected components, and partitioning vertices.

Raw-cut and normalized-cut objectives are evaluated across candidate thresholds derived from the sorted Fiedler-vector values.

## Experiments

The bridge-strength experiment constructs two dense communities and progressively adds edges between them. It examines how the Fiedler eigenvalue, normalized-cut score, and resulting partition respond to changes in graph connectivity.

## Project Structure

```text
spectral-graph-clustering/
├── src/
│   ├── __init__.py
│   └── spectral_core.py
├── experiments/
│   ├── __init__.py
│   └── bridge_strength.py
├── notebooks/
├── figures/
├── requirements.txt
└── README.md
```

## Tools

Python · NumPy · Git

## Current Limitations

The current implementation focuses on two-way partitioning of undirected, non-negatively weighted graphs without self-loops. Threshold search considers only partitions obtainable from the Fiedler vector; it does not guarantee a globally optimal cut.



