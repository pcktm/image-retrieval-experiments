from numba import njit, jit, prange
import numpy as np


@njit(fastmath=True, cache=True)
def chi_square_distance(a, b):
    return np.abs(0.5 * np.sum((a - b)**2 / (a + b + 1e-6)))


@njit(fastmath=True, cache=True)
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))


@njit(fastmath=True, cache=True)
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@njit(fastmath=True, cache=True)
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))


@njit(fastmath=True, cache=True)
def minkowski_distance(a, b, p=3):
    return np.sum(np.abs(a - b)**p)**(1 / p)


@njit(fastmath=True, cache=True)
def hamming_distance(a, b):
    return np.sum(a != b)


@njit(fastmath=True, cache=True)
def jaccard_distance(a, b):
    return 1 - np.sum(np.minimum(a, b)) / np.sum(np.maximum(a, b))


@njit(fastmath=True, cache=True)
def pearson_correlation(a, b):
    return np.corrcoef(a, b)[0, 1]


@njit(fastmath=True, cache=True)
def earth_movers_distance(a, b):
    return np.sum(np.abs(np.cumsum(a) - np.cumsum(b)))

metrics = [
    ("Euclidean", euclidean_distance),
    ("Minkowski", minkowski_distance),
    ("Cosine", cosine_distance),
]