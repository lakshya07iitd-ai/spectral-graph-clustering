
import numpy as np

A = np.array([
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
    [0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0]
], dtype=float)

degrees = A.sum(axis=1)
D = np.diag(degrees)
L = D - A

inv_sqrt = np.zeros_like(degrees)
positive = degrees > 0
inv_sqrt[positive] = 1 / np.sqrt(degrees[positive])
D_inv_sqrt = np.diag(inv_sqrt)

L_sym = D_inv_sqrt @ L @ D_inv_sqrt

ordinary_values, ordinary_vectors = np.linalg.eigh(L)
normal_values, normal_vectors = np.linalg.eigh(L_sym)

ordinary_vector = ordinary_vectors[:, 1]
normal_vector = normal_vectors[:, 1]


def split_at_zero(vector):
    group1 = np.where(vector >= 0)[0]
    group2 = np.where(vector < 0)[0]
    return group1, group2


def normalized_cut(group1, group2):
    cut = A[np.ix_(group1, group2)].sum()
    volume1 = degrees[group1].sum()
    volume2 = degrees[group2].sum()

    if volume1 == 0 or volume2 == 0:
        return float("inf")

    return cut / volume1 + cut / volume2


ordinary_groups = split_at_zero(ordinary_vector)
normal_groups = split_at_zero(normal_vector)

print("Degrees:", degrees.astype(int))

print("\nOrdinary Laplacian partition:")
print("Group 1:", ordinary_groups[0])
print("Group 2:", ordinary_groups[1])
print("Normalized cut:", round(
    normalized_cut(*ordinary_groups), 4
))

print("\nNormalized Laplacian partition:")
print("Group 1:", normal_groups[0])
print("Group 2:", normal_groups[1])
print("Normalized cut:", round(
    normalized_cut(*normal_groups), 4
))
