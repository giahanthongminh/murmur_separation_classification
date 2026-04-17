'''
CSSA-ZCR
CSSA-kurtosis
Compare methods (correlation)
Outputs:
- normal sound
- murmur
'''

import numpy as np
from ssa import ssa_decompose


def zero_crossing_rate(signal):
    """Count how often the signal changes sign."""
    if len(signal) < 2:
        return 0.0

    signs = np.sign(signal)
    return np.mean(np.diff(signs) != 0)


def cssa_zcr(signal, L, zcr_threshold=0.05):
    """
    Reconstruct normal heart sound using low-ZCR components.
    Murmur is the residual.
    """
    components = ssa_decompose(signal, L)

    selected = []
    zcr_values = []

    for i, comp in enumerate(components):
        zcr = zero_crossing_rate(comp)
        zcr_values.append(zcr)

        if zcr <= zcr_threshold:
            selected.append(i)

    if len(selected) == 0:
        normal_reconstructed = np.zeros_like(signal)
    else:
        normal_reconstructed = np.sum(components[selected], axis=0)

    murmur = signal - normal_reconstructed

    return normal_reconstructed, murmur, selected, zcr_values

def kurtosis_score(signal):
    """Measure how peaky the signal distribution is."""
    signal = np.asarray(signal)
    signal = signal - np.mean(signal)

    std = np.std(signal)
    if std == 0:
        return 0.0

    z = signal / std
    return np.mean(z ** 4) - 3


def cssa_kurtosis(signal, L, top_k=5):
    """
    Reconstruct normal heart sound using high-kurtosis components.
    Murmur is the residual.
    """
    components = ssa_decompose(signal, L)

    kurt_values = []
    for comp in components:
        kurt_values.append(kurtosis_score(comp))

    # Pick the top-k most peaky components
    ranked = np.argsort(kurt_values)[::-1]
    selected = ranked[:top_k]

    normal_reconstructed = np.sum(components[selected], axis=0)
    murmur = signal - normal_reconstructed

    return normal_reconstructed, murmur, selected, kurt_values

def correlation_score(x, y):
    """Measure overlap between reconstructed normal sound and murmur."""
    x = np.asarray(x)
    y = np.asarray(y)

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


def compare_cssa_methods(signal, L, zcr_threshold=0.05, top_k=5):
    """Run both CSSA methods and keep the one with lower correlation."""
    normal_zcr, murmur_zcr, selected_zcr, zcr_values = cssa_zcr(
        signal, L, zcr_threshold=zcr_threshold
    )
    normal_kurt, murmur_kurt, selected_kurt, kurt_values = cssa_kurtosis(
        signal, L, top_k=top_k
    )

    corr_zcr = correlation_score(normal_zcr, murmur_zcr)
    corr_kurt = correlation_score(normal_kurt, murmur_kurt)

    if abs(corr_zcr) <= abs(corr_kurt):
        best_method = "zcr"
        best_normal = normal_zcr
        best_murmur = murmur_zcr
    else:
        best_method = "kurtosis"
        best_normal = normal_kurt
        best_murmur = murmur_kurt

    return {
        "best_method": best_method,
        "best_normal": best_normal,
        "best_murmur": best_murmur,
        "corr_zcr": corr_zcr,
        "corr_kurt": corr_kurt,
        "selected_zcr": selected_zcr,
        "selected_kurt": selected_kurt,
    }