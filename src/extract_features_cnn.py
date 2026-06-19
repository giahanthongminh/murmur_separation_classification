# extract_features_cnn.py
# Extract Mel Spectrogram from separated murmur WAV files
# Saves as numpy arrays for CNN input

import pandas as pd
import numpy as np
import librosa
from pathlib import Path

DATA_ROOT = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
labels_path  = DATA_ROOT / "labels.csv"
output_dir   = DATA_ROOT / "output"
spectrogram_dir = DATA_ROOT / "spectrograms"
spectrogram_dir.mkdir(exist_ok=True)

labels = pd.read_csv(labels_path)
rows = []

for _, row in labels.iterrows():
    patient_id = row["Patient ID"]
    label = row["Systolic murmur timing"]

    for location in ["AV", "PV", "TV", "MV"]:
        wav_path = output_dir / f"{patient_id}_{location}" / "murmur_separated.wav"
        if not wav_path.exists():
            continue

        signal, sr = librosa.load(wav_path, sr=4000)

        # Mel spectrogram — 64 mel bands, fixed size 64×64
        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=64, n_fft=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Resize to fixed 64×64 by cropping or padding time axis
        target_len = 64
        if mel_db.shape[1] >= target_len:
            mel_db = mel_db[:, :target_len]
        else:
            pad = target_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")

        # Save spectrogram as .npy
        key = f"{patient_id}_{location}"
        np.save(spectrogram_dir / f"{key}.npy", mel_db)

        rows.append({"key": key, "patient_id": patient_id, "label": label})

df = pd.DataFrame(rows)
df.to_csv(DATA_ROOT / "labels_cnn.csv", index=False)
print(f"Total spectrograms: {len(df)}")
print(df["label"].value_counts())