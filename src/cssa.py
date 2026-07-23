'''
CSSA-ZCR
CSSA-kurtosis (GA-optimized joint selection)
Compare methods (correlation)
Outputs:
- normal sound
- murmur

Implements the CSSA stage of Qi & Sanei, "Murmur Separation and Classification
from Heart Sound Using Constrained Singular Spectrum Analysis and Wavelet
Transform," APSIPA ASC 2024.
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


def cssa_kurtosis(signal, L, pop_size=30, n_generations=40,
                   mutation_rate=0.05, random_state=42):
    """
    Reconstruct normal heart sound by selecting the subset of SSA components
    W in {0,1}^d that maximizes kurtosis(R @ W).

    This is a nonlinear integer programming problem (the objective depends on
    the *combined* reconstructed signal, not on each component independently),
    so it's solved with a Genetic Algorithm rather than ranking components by
    their individual kurtosis.
    Murmur is the residual.
    """
    components = ssa_decompose(signal, L)
    d = components.shape[0]
    rng = np.random.default_rng(random_state)

    kurt_values = [kurtosis_score(comp) for comp in components]

    def fitness(w):
        if not w.any():
            return -np.inf
        return kurtosis_score(w @ components)

    population = rng.integers(0, 2, size=(pop_size, d)).astype(np.int8)
    best_w, best_fit = None, -np.inf

    for _ in range(n_generations):
        fitness_vals = np.array([fitness(w) for w in population])

        gen_best_idx = np.argmax(fitness_vals)
        if fitness_vals[gen_best_idx] > best_fit:
            best_fit = fitness_vals[gen_best_idx]
            best_w = population[gen_best_idx].copy()

        # Elitism + tournament selection + uniform crossover + mutation
        next_population = [best_w.copy()]
        while len(next_population) < pop_size:
            i, j = rng.integers(0, pop_size, size=2)
            parent1 = population[i] if fitness_vals[i] > fitness_vals[j] else population[j]
            i, j = rng.integers(0, pop_size, size=2)
            parent2 = population[i] if fitness_vals[i] > fitness_vals[j] else population[j]

            mask = rng.integers(0, 2, size=d).astype(bool)
            child = np.where(mask, parent1, parent2)

            flip = rng.random(d) < mutation_rate
            child = np.where(flip, 1 - child, child).astype(np.int8)

            next_population.append(child)

        population = np.array(next_population[:pop_size])

    if best_w is None or not best_w.any():
        normal_reconstructed = np.zeros_like(signal)
        selected = np.array([], dtype=int)
    else:
        normal_reconstructed = best_w @ components
        selected = np.flatnonzero(best_w)

    murmur = signal - normal_reconstructed

    return normal_reconstructed, murmur, selected, kurt_values

def correlation_score(x, y):
    """Measure overlap between reconstructed normal sound and murmur."""
    x = np.asarray(x)
    y = np.asarray(y)

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


def compare_cssa_methods(signal, L, zcr_threshold=0.05):
    """Run both CSSA methods and keep the one with lower correlation."""
    normal_zcr, murmur_zcr, selected_zcr, zcr_values = cssa_zcr(
        signal, L, zcr_threshold=zcr_threshold
    )
    normal_kurt, murmur_kurt, selected_kurt, kurt_values = cssa_kurtosis(signal, L)

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