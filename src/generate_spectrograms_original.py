# generate_spectrograms_original.py
#
# Same as generate_spectrograms_dl.py but uses ORIGINAL WAV files (no separation)
# — baseline to compare against separated murmur pipeline
#
# Output:
#   spectrograms_original/<patient>_<loc>_seg_<i>.npy  (3, 224, 224)
#   labels_original.csv

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from PIL import Image

DATA_ROOT   = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
wav_dir     = DATA_ROOT / "training_data"
tsv_dir     = DATA_ROOT / "training_data"
spec_dir    = DATA_ROOT / "spectrograms_original"
labels_path = DATA_ROOT / "labels.csv"
output_csv  = DATA_ROOT / "labels_original.csv"

spec_dir.mkdir(exist_ok=True)

SR     = 4000
N_MELS = 128
N_FFT  = 512
HOP    = 128
IMG_H  = 224
IMG_W  = 224


def mel_spectrogram(signal, sr):
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    n_fft  = min(N_FFT, len(signal))
    if n_fft < 2:
        n_fft = 2
    elif n_fft % 2 != 0:
        n_fft -= 1
    hop = min(HOP, n_fft // 2)
    mel    = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=N_MELS, n_fft=n_fft, hop_length=hop)
    return librosa.power_to_db(mel, ref=np.max)


def to_image(mel_db):
    lo, hi = mel_db.min(), mel_db.max()
    arr = np.zeros((IMG_H, IMG_W), dtype=np.float32) if hi - lo < 1e-6 \
          else (mel_db - lo) / (hi - lo)
    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    pil = pil.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return np.stack([arr, arr, arr], axis=0)   # (3, H, W)


def imagenet_normalize(arr3):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (arr3 - mean) / std


labels_df = pd.read_csv(labels_path)
label_map = dict(zip(labels_df["Patient ID"].astype(str),
                     labels_df["Systolic murmur timing"]))

rows = []

for _, row in labels_df.iterrows():
    patient_id = str(row["Patient ID"])
    label      = row["Systolic murmur timing"]

    for location in ["AV", "PV", "TV", "MV"]:
        wav_path = wav_dir / f"{patient_id}_{location}.wav"
        tsv_path = tsv_dir / f"{patient_id}_{location}.tsv"

        if not wav_path.exists() or not tsv_path.exists():
            continue

        signal, sr = librosa.load(wav_path, sr=SR)

        tsv     = pd.read_csv(tsv_path, sep="\t", header=None,
                               names=["start", "end", "label"])
        systole = tsv[tsv["label"] == 2]

        for seg_idx, (_, seg_row) in enumerate(systole.iterrows()):
            s = int(seg_row["start"] * sr)
            e = int(seg_row["end"]   * sr)
            seg = signal[s:e]

            if len(seg) < 64:
                continue

            mel_db = mel_spectrogram(seg, SR)
            arr3   = imagenet_normalize(to_image(mel_db))

            key = f"{patient_id}_{location}_seg_{seg_idx}"
            np.save(spec_dir / f"{key}.npy", arr3)
            rows.append({"key": key, "patient_id": patient_id,
                         "location": location, "seg_idx": seg_idx,
                         "label": label})

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} spectrograms → {spec_dir}")
print(df["label"].value_counts())
