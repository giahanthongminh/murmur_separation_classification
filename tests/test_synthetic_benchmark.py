import numpy as np
import pandas as pd

from src.evaluation.synthetic_benchmark import (
    _select_tuning_configuration,
    make_synthetic_mixture,
)


def test_synthetic_cycle_has_disjoint_complete_phases_and_systolic_murmur() -> None:
    sample = make_synthetic_mixture(
        "early_systolic",
        sample_rate=1000,
        duration=0.8,
        murmur_to_heart_db=-3.0,
        snr_db=20.0,
        seed=42,
    )
    masks = sample["phase_masks"]
    assert isinstance(masks, dict)
    coverage = np.sum(np.stack(list(masks.values())), axis=0)
    assert np.all(coverage == 1)
    murmur = np.asarray(sample["true_murmur"])
    systole = np.asarray(masks["systole"], dtype=bool)
    assert np.any(np.abs(murmur[systole]) > 0)
    assert np.allclose(murmur[~systole], 0)
    assert sample["murmur_present"] is True
    assert 0 <= float(sample["onset_seconds"]) < float(sample["offset_seconds"])


def test_synthetic_absent_control_has_no_murmur_source() -> None:
    sample = make_synthetic_mixture(
        "absent",
        sample_rate=1000,
        duration=0.8,
        murmur_to_heart_db=0.0,
        snr_db=20.0,
        seed=42,
    )
    assert not np.any(np.asarray(sample["true_murmur"]))
    assert sample["murmur_present"] is False
    assert sample["onset_seconds"] is None
    assert sample["offset_seconds"] is None


def test_tuning_selection_uses_near_best_leakage_tie_breakers() -> None:
    summary = pd.DataFrame(
        [
            {
                "minimum_systole_focus": 0.06,
                "minimum_systole_to_s1_s2_ratio": 0.10,
                "ssa_window_ms": 20.0,
                "mean_si_sdr_murmur": 10.0,
                "present_fallback_rate": 0.5,
                "absent_candidate_energy_ratio": 0.2,
                "mean_normal_leakage_into_murmur": 0.2,
                "mean_outside_murmur_energy_ratio": 0.2,
                "mean_murmur_region_energy_retention": 0.8,
            },
            {
                "minimum_systole_focus": 0.12,
                "minimum_systole_to_s1_s2_ratio": 0.20,
                "ssa_window_ms": 25.0,
                "mean_si_sdr_murmur": 9.7,
                "present_fallback_rate": 0.0,
                "absent_candidate_energy_ratio": 0.05,
                "mean_normal_leakage_into_murmur": 0.05,
                "mean_outside_murmur_energy_ratio": 0.05,
                "mean_murmur_region_energy_retention": 0.7,
            },
            {
                "minimum_systole_focus": 0.16,
                "minimum_systole_to_s1_s2_ratio": 0.30,
                "ssa_window_ms": 35.0,
                "mean_si_sdr_murmur": 9.0,
                "present_fallback_rate": 0.0,
                "absent_candidate_energy_ratio": 0.01,
                "mean_normal_leakage_into_murmur": 0.01,
                "mean_outside_murmur_energy_ratio": 0.01,
                "mean_murmur_region_energy_retention": 0.6,
            },
        ]
    )
    selected = _select_tuning_configuration(summary)
    assert selected["minimum_systole_focus"] == 0.12
    assert selected["minimum_systole_to_s1_s2_ratio"] == 0.20
    assert selected["ssa_window_ms"] == 25.0
