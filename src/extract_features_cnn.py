"""Create 64x64 mel features from project-owned murmur candidates."""

import numpy as np
import pandas as pd
import librosa

from config import FEATURE_OUTPUT_DIR, LABELS_PATH, SEPARATION_OUTPUT_DIR
from src.data_validation import validate_dataset


def main() -> int:
    validate_dataset()
    spectrogram_dir = FEATURE_OUTPUT_DIR / "spectrograms"
    spectrogram_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_PATH, dtype={"Patient ID": str})
    label_map = dict(
        zip(labels["Patient ID"], labels["Systolic murmur timing"], strict=True)
    )
    rows = []
    for recording_dir in sorted(SEPARATION_OUTPUT_DIR.iterdir()):
        if not recording_dir.is_dir() or "_" not in recording_dir.name:
            continue
        patient_id, location = recording_dir.name.rsplit("_", 1)
        if patient_id not in label_map:
            continue
        candidates = sorted(recording_dir.glob("seg_*_murmur_candidate.npy"))
        for path in candidates:
            segment_index = path.stem.split("_")[1]
            signal = np.load(path).astype(float)
            if len(signal) < 16:
                continue
            n_fft = min(512, len(signal))
            n_fft -= n_fft % 2
            mel = librosa.feature.melspectrogram(
                y=signal, sr=4000, n_mels=64, n_fft=max(2, n_fft)
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            target = 64
            if mel_db.shape[1] >= target:
                mel_db = mel_db[:, :target]
            else:
                mel_db = np.pad(
                    mel_db, ((0, 0), (0, target - mel_db.shape[1])), mode="constant"
                )
            key = f"{recording_dir.name}_seg_{segment_index}"
            np.save(spectrogram_dir / f"{key}.npy", mel_db.astype(np.float32))
            rows.append(
                {
                    "key": key,
                    "patient_id": patient_id,
                    "location": location,
                    "segment_index": int(segment_index),
                    "label": label_map[patient_id],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(FEATURE_OUTPUT_DIR / "labels_cnn.csv", index=False)
    print(f"Total spectrograms: {len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
