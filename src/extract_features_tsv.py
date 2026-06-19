# extract_features_tsv.py
# Extract MFCC features from systolic segments only (using TSV segmentation)
# Cleaner than full audio — focuses only on the period where murmur occurs

import pandas as pd
import numpy as np
import librosa
from pathlib import Path

DATA_ROOT = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
labels_path   = DATA_ROOT / "labels.csv"
output_dir    = DATA_ROOT / "output"
tsv_dir       = DATA_ROOT / "training_data"   # TSV files alongside WAVs
features_path = DATA_ROOT / "features_tsv.csv"

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

        signal, sr = librosa.load(wav_path, sr=4000)

        # Load segmentation — label=2 marks systolic periods
        tsv = pd.read_csv(tsv_path, sep="\t", header=None,
                          names=["start", "end", "label"])
        systole = tsv[tsv["label"] == 2]

        if len(systole) == 0:
            continue

        # Concatenate all systolic segments into one signal
        segments = [signal[int(s*sr):int(e*sr)]
                    for _, (s, e, _) in systole.iterrows()]
        systole_signal = np.concatenate(segments)

        if len(systole_signal) < 100:
            continue

        # 40 MFCC coefficients, averaged over time
        mfcc = librosa.feature.mfcc(y=systole_signal, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)

        row_data = {"patient_id": patient_id, "label": label}
        for i, val in enumerate(mfcc_mean):
            row_data[f"mfcc_{i}"] = val
        rows.append(row_data)

df = pd.DataFrame(rows)
df.to_csv(features_path, index=False)
print(f"Total: {len(df)}")
print(df["label"].value_counts())