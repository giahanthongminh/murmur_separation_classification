# extract_features_rich.py
# Extract rich temporal + spectral features from separated murmur segments
# Key addition: energy profile across systole segments — directly encodes timing
#
# Reads per-segment murmur arrays produced by run_separation_per_segment.py
# (CSSA+DWT run on each systolic segment individually, preserving inter-beat
# timing), rather than a whole-recording separation.

import json
import pandas as pd
import numpy as np
import librosa
from scipy.stats import skew, kurtosis

from config import FEATURE_OUTPUT_DIR, LABELS_PATH, SEPARATION_OUTPUT_DIR
from src.data_validation import validate_dataset

labels_path = LABELS_PATH
output_dir = SEPARATION_OUTPUT_DIR
features_path = FEATURE_OUTPUT_DIR / "features_rich.csv"

SR = 4000
N_ENERGY_SEGMENTS = 10  # split systole into 10 segments → RMS per segment


def extract_features(signal, sr):
    """
    Extract ~187 features from a signal:
    - MFCC mean + std (80)
    - Delta MFCC mean + std (80)
    - Energy profile: RMS per 10 systole segments (10)
    - Energy stats: centroid, first/second half ratio, skew, kurtosis, max pos, std (6)
    - Spectral: centroid, rolloff, bandwidth, ZCR mean+std each (8)
    - Onset strength: mean, max, time-of-max (3)
    """
    features = {}

    # --- MFCC mean + std ---
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    for i in range(40):
        features[f"mfcc_mean_{i}"] = np.mean(mfcc[i])
        features[f"mfcc_std_{i}"]  = np.std(mfcc[i])

    # --- Delta MFCC mean + std ---
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
    width = max(width, 3)
    if mfcc.shape[1] >= 3:
        delta = librosa.feature.delta(mfcc, width=width)
        for i in range(40):
            features[f"delta_mean_{i}"] = np.mean(delta[i])
            features[f"delta_std_{i}"]  = np.std(delta[i])
    else:
        for i in range(40):
            features[f"delta_mean_{i}"] = 0.0
            features[f"delta_std_{i}"]  = 0.0

    # --- Energy profile: RMS per segment (key timing feature) ---
    seg_len = max(1, len(signal) // N_ENERGY_SEGMENTS)
    rms_profile = []
    for i in range(N_ENERGY_SEGMENTS):
        seg = signal[i*seg_len:(i+1)*seg_len]
        rms_profile.append(np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0.0)
    rms_profile = np.array(rms_profile)
    for i, val in enumerate(rms_profile):
        features[f"energy_seg_{i}"] = val

    # --- Energy stats ---
    total = rms_profile.sum() + 1e-8
    weights = np.arange(N_ENERGY_SEGMENTS)
    features["energy_centroid"]   = np.sum(weights * rms_profile) / total
    first_half  = rms_profile[:N_ENERGY_SEGMENTS//2].sum()
    second_half = rms_profile[N_ENERGY_SEGMENTS//2:].sum()
    features["energy_half_ratio"] = first_half / (second_half + 1e-8)
    features["energy_skew"]       = float(skew(rms_profile))
    features["energy_kurtosis"]   = float(kurtosis(rms_profile))
    features["energy_max_pos"]    = float(np.argmax(rms_profile))
    features["energy_std"]        = float(np.std(rms_profile))

    # --- Spectral features ---
    spec_centroid  = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spec_rolloff   = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    zcr            = librosa.feature.zero_crossing_rate(signal)[0]

    for name, feat in [("spec_centroid", spec_centroid),
                       ("spec_rolloff",  spec_rolloff),
                       ("spec_bandwidth",spec_bandwidth),
                       ("zcr",           zcr)]:
        features[f"{name}_mean"] = np.mean(feat)
        features[f"{name}_std"]  = np.std(feat)

    # --- Onset strength ---
    onset = librosa.onset.onset_strength(y=signal, sr=sr)
    features["onset_mean"]    = np.mean(onset)
    features["onset_max"]     = np.max(onset)
    features["onset_time_max"] = float(np.argmax(onset)) / (len(onset) + 1e-8)

    return features


def load_systole_signal(patient_id, location):
    """Concatenate per-segment separated murmur arrays in timing order."""
    folder = output_dir / f"{patient_id}_{location}"
    meta_path = folder / "segments_meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    segments = []
    for entry in sorted(meta, key=lambda m: m.get("cycle_index", m.get("seg_idx", 0))):
        segment_index = entry.get("cycle_index", entry.get("seg_idx", 0))
        seg_path = folder / f"seg_{segment_index}_murmur_candidate.npy"
        if not seg_path.exists():
            seg_path = folder / f"seg_{segment_index}_murmur.npy"
        if seg_path.exists():
            segments.append(np.load(seg_path))

    return np.concatenate(segments) if segments else None


validate_dataset()
labels = pd.read_csv(labels_path)
rows = []

for _, row in labels.iterrows():
    patient_id = row["Patient ID"]
    label = row["Systolic murmur timing"]

    for location in ["AV", "PV", "TV", "MV"]:
        systole_signal = load_systole_signal(patient_id, location)

        if systole_signal is None or len(systole_signal) < 100:
            continue

        feats = extract_features(systole_signal, SR)
        feats["patient_id"] = patient_id
        feats["label"] = label
        rows.append(feats)

df = pd.DataFrame(rows)
df.to_csv(features_path, index=False)
print(f"Total: {len(df)}")
print(df["label"].value_counts())
