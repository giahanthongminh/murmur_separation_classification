# ssa.py
# Singular Spectrum Analysis (SSA) decomposition
# Decomposes a 1D signal into components via SVD of the trajectory matrix

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def embed_signal(signal, L):
    """
    Build L×K trajectory (Hankel) matrix using zero-copy sliding windows.
    L = window length, K = N - L + 1
    """
    return sliding_window_view(signal, L).T


def diagonal_averaging(X):
    """
    Convert SSA component matrix back to 1D signal by averaging anti-diagonals.
    Vectorized: O(L) Python iterations instead of O(L×K).
    """
    L, K = X.shape
    N = L + K - 1
    result = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        result[i:i+K] += X[i]   # vectorized numpy add across row i
        counts[i:i+K] += 1
    return result / counts


def ssa_decompose(signal, L, n_components=20):
    """
    Decompose signal into SSA components.
    Only reconstructs n_components (default 20) to avoid unnecessary work —
    CSSA only needs the top few components to separate heart sound from murmur.
    """
    X = embed_signal(signal, L)

    # SVD of trajectory matrix
    U, S, VT = np.linalg.svd(X, full_matrices=False)

    k = n_components or len(S)
    components = []
    for i in range(k):
        # Rank-1 reconstruction for component i
        Xi = S[i] * np.outer(U[:, i], VT[i, :])
        components.append(diagonal_averaging(Xi))

    return np.array(components)