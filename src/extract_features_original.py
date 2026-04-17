# extract_features_original.py
# Same as extract_features.py but uses ORIGINAL wav files
# This is for comparison: original vs separated murmur

import pandas as pd
import numpy as np
import librosa
from pathlib import Path

# Paths
labels_path = Path("/Users/danggiahan/Documents/heart_sounds/labels.csv")
wav_dir = Path("/Users/danggiahan/Documents/heart_sounds/wav")
features_path = Path("/Users/danggiahan/Documents/heart_sounds/features_original.csv")

# Load labels
labels = pd.read_csv(labels_path)

rows = []

for _, row in labels.iterrows():
    patient_id = row["Patient ID"]
    label = row["Systolic murmur timing"]

    # Use first available original wav
    found = False
    for location in ["AV", "PV", "TV", "MV"]:
        wav_path = wav_dir / f"{patient_id}_{location}.wav"
        if wav_path.exists():
            signal, sr = librosa.load(wav_path, sr=4000)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc, axis=1)

            row_data = {"patient_id": patient_id, "label": label}
            for i, val in enumerate(mfcc_mean):
                row_data[f"mfcc_{i}"] = val

            rows.append(row_data)
            found = True

    if not found:
        print(f"WARNING: No wav found for patient {patient_id}")

# Save
df = pd.DataFrame(rows)
df.to_csv(features_path, index=False)

print(f"Total features extracted: {len(df)}")
print(df["label"].value_counts())