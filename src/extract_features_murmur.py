# extract_features_murmur.py
# Extract MFCC features from separated murmur WAV files
# These are the main features used for murmur timing classification

import pandas as pd
import numpy as np
import librosa
from pathlib import Path

DATA_ROOT = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
labels_path   = DATA_ROOT / "labels.csv"
output_dir    = DATA_ROOT / "output"
features_path = DATA_ROOT / "features.csv"

labels = pd.read_csv(labels_path)
rows = []

for _, row in labels.iterrows():
    patient_id = row["Patient ID"]
    label = row["Systolic murmur timing"]

    # Use all available locations per patient
    for location in ["AV", "PV", "TV", "MV"]:
        wav_path = output_dir / f"{patient_id}_{location}" / "murmur_separated.wav"
        if not wav_path.exists():
            continue

        signal, sr = librosa.load(wav_path, sr=4000)

        # 40 MFCC coefficients, averaged over time
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)

        row_data = {"patient_id": patient_id, "label": label}
        for i, val in enumerate(mfcc_mean):
            row_data[f"mfcc_{i}"] = val
        rows.append(row_data)

df = pd.DataFrame(rows)
df.to_csv(features_path, index=False)
print(f"Total: {len(df)}")
print(df["label"].value_counts())