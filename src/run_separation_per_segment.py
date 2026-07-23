# run_separation_per_segment.py
#
# KEY DIFFERENCE vs run_all_separation.py:
#   OLD: concatenate all systolic segments → CSSA on blob → timing destroyed
#   NEW: CSSA on each systolic segment individually → timing within each
#        heartbeat is preserved → mel spectrogram can encode early/mid/late
#
# Output per recording: one .npy murmur array per systolic segment
#   output/<patient>_<loc>/seg_<i>_murmur.npy   (float32, sr=4000)
#   output/<patient>_<loc>/segments_meta.json    (start/end times)

import json
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

from cssa import compare_cssa_methods
from dwt_refine import dwt_refine

DATA_ROOT  = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
wav_dir    = DATA_ROOT / "training_data"
tsv_dir    = DATA_ROOT / "training_data"
output_dir = DATA_ROOT / "output_per_seg"
labels_path = DATA_ROOT / "labels.csv"

SR = 4000
MIN_SEG_SAMPLES = 200   # skip segments shorter than 50 ms


def get_systolic_segments(tsv_path, sr):
    tsv = pd.read_csv(tsv_path, sep="\t", header=None,
                      names=["start", "end", "label"])
    segs = tsv[tsv["label"] == 2]
    return [(r["start"], r["end"],
             int(r["start"] * sr), int(r["end"] * sr))
            for _, r in segs.iterrows()]


def separate_segment(seg_signal):
    """Run CSSA+DWT on a single systolic segment. Return murmur array."""
    if len(seg_signal) < MIN_SEG_SAMPLES:
        return None
    result = compare_cssa_methods(seg_signal, L=min(100, len(seg_signal) // 4),
                                  zcr_threshold=0.05)
    refined_normal = dwt_refine(result["best_normal"])
    # Align length (DWT waverec can add 1 sample)
    n = len(seg_signal)
    murmur = seg_signal - refined_normal[:n]
    return murmur.astype(np.float32)


def process_file(wav_path):
    stem = wav_path.stem                       # e.g. "12345_AV"
    out_folder = output_dir / stem

    if (out_folder / "segments_meta.json").exists():
        return None

    tsv_path = tsv_dir / (stem + ".tsv")
    if not tsv_path.exists():
        return None

    signal, sr = librosa.load(wav_path, sr=SR)
    segs = get_systolic_segments(tsv_path, sr)
    if not segs:
        return None

    out_folder.mkdir(parents=True, exist_ok=True)
    meta = []

    for i, (t_start, t_end, s_start, s_end) in enumerate(segs):
        seg = signal[s_start:s_end]
        murmur = separate_segment(seg)
        if murmur is None:
            continue
        np.save(out_folder / f"seg_{i}_murmur.npy", murmur)
        meta.append({"seg_idx": i, "t_start": t_start, "t_end": t_end,
                     "duration": t_end - t_start})

    if not meta:
        return None

    with open(out_folder / "segments_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done: {stem} — {len(meta)} segments")
    return stem


if __name__ == "__main__":
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_path)
    patient_ids = set(labels["Patient ID"].astype(str))
    files = [f for f in wav_dir.glob("*.wav")
             if f.stem.split("_")[0] in patient_ids]

    print(f"Processing {len(files)} files, {SR} Hz, per-segment CSSA")

    with Pool(processes=8) as pool:
        pool.map(process_file, files)

    print("Done. Output:", output_dir)
