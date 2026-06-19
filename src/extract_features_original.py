# extract_features_original.py
# Extract MFCC features from original (unprocessed) WAV files
# Used as baseline for comparison against separated murmur features

import pandas as pd
import numpy as np
import librosa
from pathlib import Path

DATA_ROOT = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
labels_path   = DATA_ROOT / "labels.csv"
wav_dir       = DATA_ROOT / "training_data"
features_path = DATA_ROOT / "features_original.csv"

labels = pd.read_csv(labels_path)
rows = []

for _, row in labels.iterrows():
    patient_id = row["Patient ID"]
    label = row["Systolic murmur timing"]

    # Use all available auscultation locations per patient
    for location in ["AV", "PV", "TV", "MV"]:
        wav_path = wav_dir / f"{patient_id}_{location}.wav"
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