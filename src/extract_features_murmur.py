'''
For each Patient ID in labels.csv
→ Load murmur_separated.wav from output folder
→ Extract MFCC (40 numbers)
→ Save as features.csv ready for classifier
'''

# extract_features_murmur.py
# Load separated murmur WAV files and extract MFCC features
# Output: features.csv with MFCC + label for each patient

import pandas as pd
import numpy as np
import librosa
from pathlib import Path

# Paths
labels_path = Path("/Users/danggiahan/Documents/heart_sounds/labels.csv")
output_dir = Path("/Users/danggiahan/Documents/heart_sounds/output")
features_path = Path("/Users/danggiahan/Documents/heart_sounds/features.csv")

# Load labels
labels = pd.read_csv(labels_path)

rows = []

for _, row in labels.iterrows():
    patient_id = row["Patient ID"]
    label = row["Systolic murmur timing"]

    # Each patient has multiple locations (AV, PV, TV, MV)
    # Use the first available murmur_separated.wav
    found = False
    for location in ["AV", "PV", "TV", "MV"]:
        wav_path = output_dir / f"{patient_id}_{location}" / "murmur_separated.wav"
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