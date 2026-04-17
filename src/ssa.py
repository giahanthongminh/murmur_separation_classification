'''
ssa:
1. ssa decomposition
2. turn signal into components:

- component 0:
+ usually strongest pattern
+ often main heart rhythm (S1/S2)

- component 1,2:
+ similar patterns but slightly different
+ can capture: repeated beats + variations + noise / murmur parts
'''
import numpy as np

def embed_signal(signal, L):
    """
    Turn 1D signal into a matrix of overlapping windows.

    Each column is a shifted version of the signal.
    """
    N = len(signal)
    K = N - L + 1

    # Stack sliding windows column-wise
    return np.column_stack([signal[i:i+L] for i in range(K)])


def diagonal_averaging(X):
    """
    Convert SSA matrix back into a 1D signal.

    This averages along diagonals to reconstruct time series.
    """
    L, K = X.shape
    N = L + K - 1

    result = np.zeros(N)
    count = np.zeros(N)

    # Accumulate values along diagonals
    for i in range(L):
        for j in range(K):
            result[i + j] += X[i, j]
            count[i + j] += 1

    # Average overlapping contributions
    return result / count


def ssa_decompose(signal, L):
    """Decompose signal into SSA components."""

    print("Building trajectory matrix...")
    X = embed_signal(signal, L)

    print("Running SVD...")
    U, S, VT = np.linalg.svd(X, full_matrices=False)

    print("Reconstructing components...")
    components = []

    for i in range(len(S)):
        Xi = S[i] * np.outer(U[:, i], VT[i, :])
        comp = diagonal_averaging(Xi)
        components.append(comp)

    print("Done.")
    return np.array(components)