# extract_features_rich_original.py
# Same rich features but from original WAV — for fair baseline comparison

import pandas as pd
import numpy as np
import librosa
from scipy.stats import skew, kurtosis

from config import AUDIO_DIR, FEATURE_OUTPUT_DIR, LABELS_PATH
from src.data_validation import validate_dataset

labels_path = LABELS_PATH
wav_dir = AUDIO_DIR
tsv_dir = AUDIO_DIR
features_path = FEATURE_OUTPUT_DIR / "features_rich_original.csv"

N_ENERGY_SEGMENTS = 10


def extract_features(signal, sr):
    features = {}

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    for i in range(40):
        features[f"mfcc_mean_{i}"] = np.mean(mfcc[i])
        features[f"mfcc_std_{i}"]  = np.std(mfcc[i])

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

    seg_len = max(1, len(signal) // N_ENERGY_SEGMENTS)
    rms_profile = []
    for i in range(N_ENERGY_SEGMENTS):
        seg = signal[i*seg_len:(i+1)*seg_len]
        rms_profile.append(np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0.0)
    rms_profile = np.array(rms_profile)
    for i, val in enumerate(rms_profile):
        features[f"energy_seg_{i}"] = val

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

    onset = librosa.onset.onset_strength(y=signal, sr=sr)
    features["onset_mean"]     = np.mean(onset)
    features["onset_max"]      = np.max(onset)
    features["onset_time_max"] = float(np.argmax(onset)) / (len(onset) + 1e-8)

    return features


validate_dataset()
labels = pd.read_csv(labels_path)
rows = []

for _, row in labels.iterrows():
    patient_id = row["Patient ID"]
    label = row["Systolic murmur timing"]

    for location in ["AV", "PV", "TV", "MV"]:
        wav_path = wav_dir / f"{patient_id}_{location}.wav"
        tsv_path = tsv_dir / f"{patient_id}_{location}.tsv"

        if not wav_path.exists() or not tsv_path.exists():
            continue

        signal, sr = librosa.load(wav_path, sr=4000)

        # Crop to systolic segments for fair comparison
        tsv = pd.read_csv(tsv_path, sep="\t", header=None,
                          names=["start", "end", "label"])
        systole = tsv[tsv["label"] == 2]

        if len(systole) == 0:
            continue

        segments = [signal[int(s*sr):int(e*sr)]
                    for _, (s, e, _) in systole.iterrows()]
        systole_signal = np.concatenate(segments)

        if len(systole_signal) < 100:
            continue

        feats = extract_features(systole_signal, sr)
        feats["patient_id"] = patient_id
        feats["label"] = label
        rows.append(feats)

df = pd.DataFrame(rows)
df.to_csv(features_path, index=False)
print(f"Total: {len(df)}")
print(df["label"].value_counts())
