# generate_spectrograms_dl.py
#
# Converts per-segment murmur .npy files → mel spectrogram images
# saved as (3, 224, 224) float32 .npy, ready for ImageNet-pretrained CNNs.
#
# Timing is preserved because each segment keeps its internal structure:
#   - Early-systolic murmur → high energy at beginning of spectrogram
#   - Mid-systolic         → high energy in the middle
#   - Holosystolic         → energy spread across the whole image
#
# Output:
#   spectrograms_dl/<patient>_<loc>_seg_<i>.npy   (3, 224, 224)
#   labels_dl.csv  — one row per segment with patient_id, label, key

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from PIL import Image

DATA_ROOT     = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
seg_dir       = DATA_ROOT / "output_per_seg"
spec_dir      = DATA_ROOT / "spectrograms_dl"
labels_path   = DATA_ROOT / "labels.csv"
output_csv    = DATA_ROOT / "labels_dl.csv"

spec_dir.mkdir(exist_ok=True)

SR      = 4000
N_MELS  = 128
N_FFT   = 512
HOP     = 128
IMG_H   = 224
IMG_W   = 224

# ImageNet normalization constants (will be applied during training, not here)
# Here we only save the raw mel-dB image in [0,1]


def mel_spectrogram(signal, sr):
    # Replace NaN/Inf from imperfect CSSA+DWT separation
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    # n_fft must not exceed signal length
    n_fft = min(N_FFT, len(signal))
    # n_fft must be even and at least 2
    if n_fft < 2:
        n_fft = 2
    elif n_fft % 2 != 0:
        n_fft -= 1
    hop = min(HOP, n_fft // 2)
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=N_MELS, n_fft=n_fft, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db   # shape: (N_MELS, T)


def to_image(mel_db):
    """Normalize mel-dB to [0,1] and resize to (IMG_H, IMG_W)."""
    lo, hi = mel_db.min(), mel_db.max()
    if hi - lo < 1e-6:
        arr = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    else:
        arr = (mel_db - lo) / (hi - lo)

    # PIL resize: input must be (H, W) uint8
    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    pil = pil.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0  # back to [0,1]

    # Repeat to 3 channels so pretrained RGB models work directly
    arr3 = np.stack([arr, arr, arr], axis=0)   # (3, H, W)
    return arr3


def imagenet_normalize(arr3):
    """Apply ImageNet mean/std normalization channel-wise."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (arr3 - mean) / std


labels_df = pd.read_csv(labels_path)
label_map  = dict(zip(labels_df["Patient ID"].astype(str),
                      labels_df["Systolic murmur timing"]))

rows = []

for seg_folder in sorted(seg_dir.iterdir()):
    if not seg_folder.is_dir():
        continue

    stem = seg_folder.name          # e.g. "12345_AV"
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        continue
    patient_id, location = parts[0], parts[1]

    if patient_id not in label_map:
        continue
    label = label_map[patient_id]

    for npy_path in sorted(seg_folder.glob("seg_*_murmur.npy")):
        seg_idx = npy_path.stem.split("_")[1]   # "0", "1", ...
        signal  = np.load(npy_path).astype(np.float32)

        if len(signal) < 64:
            continue

        if not np.all(np.isfinite(signal)):
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        mel_db = mel_spectrogram(signal, SR)
        arr3   = to_image(mel_db)
        arr3   = imagenet_normalize(arr3)

        key = f"{stem}_seg_{seg_idx}"
        np.save(spec_dir / f"{key}.npy", arr3)

        rows.append({"key": key, "patient_id": patient_id,
                     "location": location, "seg_idx": int(seg_idx),
                     "label": label})

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} spectrograms → {spec_dir}")
print(df["label"].value_counts())
