# feature_utils.py
# Shared feature extraction functions for CirCor pipeline.
# All three conditions (Original WAV, Separated Murmur, TSV Systole) use the same
# functions here to guarantee identical feature spaces across CSVs.

import numpy as np
import librosa
from scipy.signal import welch, hilbert, find_peaks
from scipy.stats import skew, kurtosis, entropy as sp_entropy


# ── MFCC (mean + std) + Delta MFCC (mean + std) ─────────────────────────────

def extract_mfcc_features(signal, sr, n_mfcc=40):
    """160 features: mfcc_mean_0..39, mfcc_std_0..39, delta_mean_0..39, delta_std_0..39"""
    features = {}

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f"mfcc_mean_{i}"] = float(np.mean(mfcc[i]))
        features[f"mfcc_std_{i}"]  = float(np.std(mfcc[i]))

    # Delta MFCC — width must be odd and >= 3
    width = mfcc.shape[1]
    width = width if width % 2 == 1 else width - 1
    width = max(min(width, 9), 3)

    if mfcc.shape[1] >= 3:
        delta = librosa.feature.delta(mfcc, width=width)
        for i in range(n_mfcc):
            features[f"delta_mean_{i}"] = float(np.mean(delta[i]))
            features[f"delta_std_{i}"]  = float(np.std(delta[i]))
    else:
        for i in range(n_mfcc):
            features[f"delta_mean_{i}"] = 0.0
            features[f"delta_std_{i}"]  = 0.0

    return features


# ── PSD features via Welch method ────────────────────────────────────────────

_HEART_SOUND_BANDS = [(20, 60), (60, 120), (120, 250), (250, 400), (400, 650)]


def extract_psd_features(signal, sr, bands=None):
    """
    ~17 features per band set:
      psd_band{i}_power, psd_band{i}_ratio  (2 × n_bands)
      psd_dominant_freq, psd_centroid, psd_bandwidth, psd_rolloff,
      psd_flatness, psd_entropy
    """
    if bands is None:
        bands = _HEART_SOUND_BANDS

    nperseg = min(1024, len(signal))
    freqs, psd = welch(signal, fs=sr, nperseg=nperseg)

    features = {}
    total_power = float(np.sum(psd)) + 1e-12

    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        bp = float(np.sum(psd[mask]))
        features[f"psd_band{i}_power"] = bp
        features[f"psd_band{i}_ratio"] = bp / total_power

    features["psd_dominant_freq"] = float(freqs[np.argmax(psd)])

    centroid = float(np.sum(freqs * psd) / total_power)
    features["psd_centroid"] = centroid

    features["psd_bandwidth"] = float(
        np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / total_power)
    )

    cumulative = np.cumsum(psd)
    rolloff_idx = np.searchsorted(cumulative, 0.85 * cumulative[-1])
    features["psd_rolloff"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    psd_safe = psd + 1e-12
    geo_mean = float(np.exp(np.mean(np.log(psd_safe))))
    arith_mean = float(np.mean(psd_safe))
    features["psd_flatness"] = geo_mean / arith_mean

    psd_norm = psd_safe / np.sum(psd_safe)
    features["psd_entropy"] = float(sp_entropy(psd_norm))

    return features


# ── Amplitude distribution features ─────────────────────────────────────────

def extract_amplitude_features(signal):
    """
    9 features: mean, std, variance, rms, skewness, kurtosis,
                crest_factor, amp_entropy, zcr
    """
    features = {}

    features["amp_mean"]     = float(np.mean(signal))
    features["amp_std"]      = float(np.std(signal))
    features["amp_variance"] = float(np.var(signal))

    rms = float(np.sqrt(np.mean(signal ** 2)))
    features["amp_rms"]      = rms

    features["amp_skewness"] = float(skew(signal))
    features["amp_kurtosis"] = float(kurtosis(signal))

    peak = float(np.max(np.abs(signal)))
    features["amp_crest_factor"] = peak / (rms + 1e-12)

    hist, _ = np.histogram(signal, bins=50, density=True)
    hist = hist + 1e-12
    hist = hist / np.sum(hist)
    features["amp_entropy"] = float(sp_entropy(hist))

    signs = np.sign(signal)
    signs[signs == 0] = 1
    features["amp_zcr"] = float(np.sum(signs[:-1] != signs[1:]) / len(signal))

    return features


# ── Energy profile across temporal segments ──────────────────────────────────

def extract_energy_profile(signal, n_segments=10):
    """
    n_segments + 6 features:
      energy_seg_0..N-1  (RMS per segment)
      energy_centroid, energy_half_ratio, energy_skew, energy_kurtosis,
      energy_max_pos, energy_std
    """
    features = {}

    seg_len = max(1, len(signal) // n_segments)
    profile = np.array([
        float(np.sqrt(np.mean(signal[i*seg_len:(i+1)*seg_len] ** 2)))
        if len(signal[i*seg_len:(i+1)*seg_len]) > 0 else 0.0
        for i in range(n_segments)
    ])

    for i, val in enumerate(profile):
        features[f"energy_seg_{i}"] = val

    total = profile.sum() + 1e-8
    weights = np.arange(n_segments, dtype=float)
    features["energy_centroid"]   = float(np.sum(weights * profile) / total)
    features["energy_half_ratio"] = float(
        profile[:n_segments//2].sum() / (profile[n_segments//2:].sum() + 1e-8)
    )
    features["energy_skew"]       = float(skew(profile))
    features["energy_kurtosis"]   = float(kurtosis(profile))
    features["energy_max_pos"]    = float(np.argmax(profile))
    features["energy_std"]        = float(np.std(profile))

    return features


# ── Basic spectral + onset features (librosa) ────────────────────────────────

def extract_spectral_features(signal, sr):
    """
    11 features: spectral centroid/rolloff/bandwidth/ZCR (mean+std each) + onset stats
    """
    features = {}

    for name, feat in [
        ("spec_centroid",  librosa.feature.spectral_centroid(y=signal, sr=sr)[0]),
        ("spec_rolloff",   librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]),
        ("spec_bandwidth", librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]),
        ("zcr",            librosa.feature.zero_crossing_rate(signal)[0]),
    ]:
        features[f"{name}_mean"] = float(np.mean(feat))
        features[f"{name}_std"]  = float(np.std(feat))

    onset = librosa.onset.onset_strength(y=signal, sr=sr)
    features["onset_mean"]     = float(np.mean(onset))
    features["onset_max"]      = float(np.max(onset))
    features["onset_time_max"] = float(np.argmax(onset)) / (len(onset) + 1e-8)

    return features


# ── Morphology features (per single systole cycle) ───────────────────────────

def extract_morphology_features(cycle, sr):
    """
    8 features describing the shape of the murmur envelope within one cycle.

    Hilbert envelope is used to get the amplitude modulation regardless of phase.
    Smooth before peak detection to avoid noise spikes counting as peaks.

    morph_peak_pos     : normalized position of envelope peak (0=start, 1=end)
    morph_rise_time    : fraction of cycle from 10% to 90% of peak amplitude
    morph_fall_time    : fraction of cycle from 90% back down to 10%
    morph_env_cv       : coefficient of variation of envelope (low = flat = holosystolic)
    morph_env_skew     : positive = energy concentrated late, negative = early
    morph_env_kurt     : peakedness of envelope distribution
    morph_n_peaks      : number of distinct peaks (1 = crescendo-decrescendo, >1 = complex)
    morph_symmetry     : peak_pos - 0.5 (0 = symmetric, + = late, - = early)
    """
    envelope = np.abs(hilbert(cycle))
    N = len(envelope)

    smooth_len = max(1, N // 20)
    smooth_env = np.convolve(envelope, np.ones(smooth_len) / smooth_len, mode="same")

    peak_val = float(np.max(smooth_env)) + 1e-12
    peak_pos = float(np.argmax(smooth_env)) / N

    above_10 = np.where(smooth_env >= 0.10 * peak_val)[0]
    above_90 = np.where(smooth_env >= 0.90 * peak_val)[0]

    rise_time = float(above_90[0] - above_10[0]) / N \
        if len(above_10) > 0 and len(above_90) > 0 else 0.0
    fall_time = float(above_10[-1] - above_90[-1]) / N \
        if len(above_10) > 0 and len(above_90) > 0 and above_10[-1] > above_90[-1] else 0.0

    peaks, _ = find_peaks(smooth_env,
                          height=0.3 * peak_val,
                          distance=max(1, N // 10))

    return {
        "morph_peak_pos":   peak_pos,
        "morph_rise_time":  rise_time,
        "morph_fall_time":  fall_time,
        "morph_env_cv":     float(np.std(envelope) / (np.mean(envelope) + 1e-12)),
        "morph_env_skew":   float(skew(envelope)),
        "morph_env_kurt":   float(kurtosis(envelope)),
        "morph_n_peaks":    float(len(peaks)),
        "morph_symmetry":   peak_pos - 0.5,
    }


# ── Onset timing features (per single systole cycle) ─────────────────────────

def extract_onset_timing_features(cycle, sr):
    """
    7 features describing when and how long murmur is active within one cycle.

    Uses 20 equal-length frames; a frame is "active" if its RMS > 20% of max RMS.

    onset_pos            : normalized position of first active frame (0=start, 1=end)
    onset_duration_ratio : fraction of frames that are active
    onset_peak_energy_pos: normalized position of highest-energy frame
    onset_e1_ratio       : energy fraction in first third of cycle
    onset_e2_ratio       : energy fraction in middle third
    onset_e3_ratio       : energy fraction in last third
    onset_early_vs_late  : (e1 - e3) / total  (+ve = early murmur, -ve = late murmur)
    """
    N = len(cycle)
    n_frames = 20
    frame_len = max(1, N // n_frames)

    rms_frames = np.array([
        float(np.sqrt(np.mean(cycle[i * frame_len:(i + 1) * frame_len] ** 2)))
        for i in range(n_frames)
    ])
    max_rms = float(np.max(rms_frames)) + 1e-12
    active = rms_frames >= 0.2 * max_rms

    onset_idx = int(np.argmax(active)) if active.any() else n_frames - 1
    duration  = int(np.sum(active))

    third = N // 3
    e1 = float(np.mean(cycle[:third] ** 2))
    e2 = float(np.mean(cycle[third:2 * third] ** 2))
    e3 = float(np.mean(cycle[2 * third:] ** 2))
    total_e = e1 + e2 + e3 + 1e-12

    return {
        "onset_pos":             float(onset_idx) / n_frames,
        "onset_duration_ratio":  float(duration) / n_frames,
        "onset_peak_energy_pos": float(np.argmax(rms_frames)) / n_frames,
        "onset_e1_ratio":        e1 / total_e,
        "onset_e2_ratio":        e2 / total_e,
        "onset_e3_ratio":        e3 / total_e,
        "onset_early_vs_late":   (e1 - e3) / total_e,
    }


# ── Aggregate per-cycle morphology + onset features ──────────────────────────

def aggregate_per_cycle_features(cycles, sr):
    """
    Run morphology + onset timing on each individual systole cycle,
    then return mean and std of each feature across all valid cycles.

    cycles: list of 1D numpy arrays, each one systole period
    Returns dict with keys like morph_peak_pos_mean, morph_peak_pos_std, ...
    (30 features total: 8 morph + 7 onset = 15, × mean/std = 30)
    """
    per_cycle = []
    for cyc in cycles:
        if len(cyc) < 50:
            continue
        row = {}
        row.update(extract_morphology_features(cyc, sr))
        row.update(extract_onset_timing_features(cyc, sr))
        per_cycle.append(row)

    if not per_cycle:
        keys = (list(extract_morphology_features(np.zeros(100), sr).keys()) +
                list(extract_onset_timing_features(np.zeros(100), sr).keys()))
        return {f"{k}_mean": 0.0 for k in keys} | {f"{k}_std": 0.0 for k in keys}

    aggregated = {}
    for key in per_cycle[0]:
        vals = [d[key] for d in per_cycle]
        aggregated[f"{key}_mean"] = float(np.mean(vals))
        aggregated[f"{key}_std"]  = float(np.std(vals))

    return aggregated


# ── Master function ──────────────────────────────────────────────────────────

def extract_all_features(signal, sr, cycles=None, n_mfcc=40, n_energy_segments=10):
    """
    Combines all feature groups.

    signal : concatenated systole signal (used for MFCC, PSD, amplitude, energy, spectral)
    cycles : list of individual systole cycle arrays for per-cycle morph+onset features.
             If None, falls back to treating the whole signal as one cycle (less accurate).

    Total ≈ 160 (MFCC) + 16 (PSD) + 9 (amplitude) + 16 (energy) + 11 (spectral)
          + 30 (per-cycle morph+onset) = 242 features
    """
    if cycles is None:
        cycles = [signal]

    feats = {}
    feats.update(extract_mfcc_features(signal, sr, n_mfcc=n_mfcc))
    feats.update(extract_psd_features(signal, sr))
    feats.update(extract_amplitude_features(signal))
    feats.update(extract_energy_profile(signal, n_segments=n_energy_segments))
    feats.update(extract_spectral_features(signal, sr))
    feats.update(aggregate_per_cycle_features(cycles, sr))
    return feats
