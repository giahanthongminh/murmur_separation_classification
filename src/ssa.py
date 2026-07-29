"""Auditable Singular Spectrum Analysis (SSA) decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


EPSILON = 1e-12


@dataclass(frozen=True)
class SSAResult:
    """Complete SSA reconstruction plus the configured active component count."""

    components: np.ndarray
    singular_values: np.ndarray
    active_component_count: int
    explained_energy: float
    reconstruction_error: float

    @property
    def active_components(self) -> np.ndarray:
        return self.components[: self.active_component_count]

    @property
    def remainder_components(self) -> np.ndarray:
        return self.components[self.active_component_count :]


def _validate_signal(signal: np.ndarray, window_length: int) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("SSA expects a one-dimensional signal")
    if len(values) < 3:
        raise ValueError("SSA expects at least three samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("SSA signal contains NaN or infinite values")
    if not 2 <= window_length < len(values):
        raise ValueError(
            f"window_length must be in [2, {len(values) - 1}], got {window_length}"
        )
    return values


def embed_signal(signal: np.ndarray, L: int) -> np.ndarray:
    """Build the ``L x (N-L+1)`` Hankel trajectory matrix."""

    values = _validate_signal(signal, L)
    return sliding_window_view(values, L).T


def diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
    """Map a trajectory matrix back to a signal by anti-diagonal averaging."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("diagonal_averaging expects a non-empty matrix")
    rows, columns = values.shape
    result = np.zeros(rows + columns - 1, dtype=float)
    counts = np.zeros_like(result)
    for row in range(rows):
        result[row : row + columns] += values[row]
        counts[row : row + columns] += 1
    return result / counts


def select_component_count(
    singular_values: np.ndarray,
    energy_threshold: float = 0.99,
    maximum_components: int | None = None,
) -> int:
    """Select the smallest count meeting a squared-singular-value threshold."""

    values = np.asarray(singular_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular_values must be a non-empty vector")
    if not 0 < energy_threshold <= 1:
        raise ValueError("energy_threshold must be in (0, 1]")
    energy = values**2
    total = float(energy.sum())
    count = 1 if total <= EPSILON else int(
        np.searchsorted(np.cumsum(energy) / total, energy_threshold) + 1
    )
    if maximum_components is not None:
        if maximum_components < 1:
            raise ValueError("maximum_components must be positive or None")
        count = min(count, maximum_components)
    return min(count, len(values))


def relative_reconstruction_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> float:
    original_values = np.asarray(original, dtype=float)
    reconstructed_values = np.asarray(reconstructed, dtype=float)
    if original_values.shape != reconstructed_values.shape:
        raise ValueError("reconstruction shape does not match original")
    return float(
        np.linalg.norm(original_values - reconstructed_values)
        / (np.linalg.norm(original_values) + EPSILON)
    )


def ssa_decompose_audited(
    signal: np.ndarray,
    L: int,
    *,
    energy_threshold: float = 0.99,
    maximum_components: int | None = None,
    reconstruction_tolerance: float = 1e-8,
) -> SSAResult:
    """Reconstruct every SSA component and fail if completeness is violated.

    The energy criterion decides which components remain eligible for CSSA.
    Lower-energy tail components are still reconstructed and become the explicit
    noise/artifact candidate instead of being hidden inside a residual.
    """

    values = _validate_signal(signal, L)
    trajectory = sliding_window_view(values, L).T
    left, singular_values, right_t = np.linalg.svd(trajectory, full_matrices=False)
    components = np.empty((len(singular_values), len(values)), dtype=float)
    for index, singular_value in enumerate(singular_values):
        rank_one = singular_value * np.outer(left[:, index], right_t[index, :])
        components[index] = diagonal_averaging(rank_one)

    reconstructed = components.sum(axis=0)
    reconstruction_error = relative_reconstruction_error(values, reconstructed)
    if reconstruction_error > reconstruction_tolerance:
        raise RuntimeError(
            "SSA reconstruction audit failed: "
            f"relative error {reconstruction_error:.3e} exceeds "
            f"{reconstruction_tolerance:.3e}"
        )

    active_count = select_component_count(
        singular_values, energy_threshold, maximum_components
    )
    energy = singular_values**2
    explained = float(energy[:active_count].sum() / (energy.sum() + EPSILON))
    return SSAResult(
        components=components,
        singular_values=singular_values,
        active_component_count=active_count,
        explained_energy=explained,
        reconstruction_error=reconstruction_error,
    )


def ssa_decompose(
    signal: np.ndarray,
    L: int,
    n_components: int | None = None,
    *,
    energy_threshold: float = 0.99,
) -> np.ndarray:
    """Backward-compatible component-array API without a fixed default of 20."""

    result = ssa_decompose_audited(
        signal,
        L,
        energy_threshold=energy_threshold,
        maximum_components=n_components,
    )
    return result.active_components
