# feature_utils.py
# Shared feature extraction functions for CirCor pipeline.
# All three conditions (Original WAV, Separated Murmur, TSV Systole) use the same
# functions here to guarantee identical feature spaces across CSVs.

import numpy as np
import librosa
from scipy.signal import welch
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


# ── Master function ──────────────────────────────────────────────────────────

def extract_all_features(signal, sr, n_mfcc=40, n_energy_segments=10):
    """
    Combines all feature groups.
    Total ≈ 160 (MFCC) + 17 (PSD) + 9 (amplitude) + 16 (energy) + 11 (spectral) = 213 features
    """
    feats = {}
    feats.update(extract_mfcc_features(signal, sr, n_mfcc=n_mfcc))
    feats.update(extract_psd_features(signal, sr))
    feats.update(extract_amplitude_features(signal))
    feats.update(extract_energy_profile(signal, n_segments=n_energy_segments))
    feats.update(extract_spectral_features(signal, sr))
    return feats
