# extract_features_tsv.py
# Extract MFCC features from systole regions only (using TSV segmentation)
# This gives cleaner features than using the full audio

import pandas as pd
import numpy as np
import librosa
from pathlib import Path

# Paths
labels_path = Path("/Users/danggiahan/Documents/heart_sounds/labels.csv")
output_dir = Path("/Users/danggiahan/Documents/heart_sounds/output")
tsv_dir = Path("/Users/danggiahan/Documents/heart_sounds/tsv")
features_path = Path("/Users/danggiahan/Documents/heart_sounds/features_tsv.csv")

# Load labels
labels = pd.read_csv(labels_path)

rows = []

for _, row in labels.iterrows():
    patient_id = row["Patient ID"]
    label = row["Systolic murmur timing"]

    for location in ["AV", "PV", "TV", "MV"]:
        wav_path = output_dir / f"{patient_id}_{location}" / "murmur_separated.wav"
        tsv_path = tsv_dir / f"{patient_id}_{location}.tsv"

        if not wav_path.exists() or not tsv_path.exists():
            continue

        # Load audio
        signal, sr = librosa.load(wav_path, sr=4000)

        # Load TSV
        tsv = pd.read_csv(tsv_path, sep="\t", header=None, names=["start", "end", "label"])

        # Extract only systole regions (label=2)
        systole_segments = tsv[tsv["label"] == 2]

        if len(systole_segments) == 0:
            continue

        # Concatenate all systole segments
        systole_audio = []
        for _, seg in systole_segments.iterrows():
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            systole_audio.append(signal[start_sample:end_sample])

        systole_signal = np.concatenate(systole_audio)

        if len(systole_signal) < 100:
            continue

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=systole_signal, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)

        row_data = {"patient_id": patient_id, "label": label}
        for i, val in enumerate(mfcc_mean):
            row_data[f"mfcc_{i}"] = val

        rows.append(row_data)

# Save
df = pd.DataFrame(rows)
df.to_csv(features_path, index=False)

print(f"Total features extracted: {len(df)}")
print(df["label"].value_counts())