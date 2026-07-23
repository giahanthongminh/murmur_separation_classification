# run_separation_per_segment.py
#
# KEY DIFFERENCE vs run_all_separation.py:
#   OLD: concatenate all systolic segments → CSSA on blob → timing destroyed
#   NEW: CSSA on each systolic segment individually → timing within each
#        heartbeat is preserved → mel spectrogram can encode early/mid/late
#
# Processes systole (TSV label 2) and diastole (TSV label 4) segments in
# parallel, each independently CSSA+DWT-separated per segment.
#
# Output per recording:
#   output/<patient>_<loc>/seg_<i>_murmur.npy            (systole, float32, sr=4000)
#   output/<patient>_<loc>/segments_meta.json            (systole start/end times)
#   output/<patient>_<loc>/diastole/seg_<i>_murmur.npy   (diastole, float32, sr=4000)
#   output/<patient>_<loc>/diastole/segments_meta.json   (diastole start/end times)
#
# Systole keeps its original flat layout (unchanged) so already-computed
# results stay valid; diastole is purely additive in its own subfolder.

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

# TSV segmentation labels: 1=S1, 2=systole, 3=S2, 4=diastole
SEGMENT_LABELS = {"systole": 2, "diastole": 4}


def get_segments_by_label(tsv_path, sr, label):
    tsv = pd.read_csv(tsv_path, sep="\t", header=None,
                      names=["start", "end", "label"])
    segs = tsv[tsv["label"] == label]
    return [(r["start"], r["end"],
             int(r["start"] * sr), int(r["end"] * sr))
            for _, r in segs.iterrows()]


def separate_segment(seg_signal):
    """Run CSSA+DWT on a single segment (systole or diastole). Return murmur array."""
    if len(seg_signal) < MIN_SEG_SAMPLES:
        return None
    result = compare_cssa_methods(seg_signal, L=min(100, len(seg_signal) // 4),
                                  zcr_threshold=0.05)
    refined_normal = dwt_refine(result["best_normal"])
    # Align length (DWT waverec can add 1 sample)
    n = len(seg_signal)
    murmur = seg_signal - refined_normal[:n]
    return murmur.astype(np.float32)


def separate_into_folder(signal, segs, folder):
    """Run CSSA+DWT on each segment, saving seg_<i>_murmur.npy +
    segments_meta.json into folder. Returns the number of segments saved."""
    folder.mkdir(parents=True, exist_ok=True)
    meta = []

    for i, (t_start, t_end, s_start, s_end) in enumerate(segs):
        seg = signal[s_start:s_end]
        murmur = separate_segment(seg)
        if murmur is None:
            continue
        np.save(folder / f"seg_{i}_murmur.npy", murmur)
        meta.append({"seg_idx": i, "t_start": t_start, "t_end": t_end,
                     "duration": t_end - t_start})

    if not meta:
        return 0

    with open(folder / "segments_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return len(meta)


def process_file(wav_path):
    stem = wav_path.stem                       # e.g. "12345_AV"
    out_folder = output_dir / stem

    tsv_path = tsv_dir / (stem + ".tsv")
    if not tsv_path.exists():
        return None

    signal = None
    done = []

    for seg_type, label in SEGMENT_LABELS.items():
        # Systole keeps its original flat layout (out_folder itself);
        # diastole is new and lives in its own subfolder.
        type_folder = out_folder if seg_type == "systole" else out_folder / seg_type

        if (type_folder / "segments_meta.json").exists():
            continue

        if signal is None:
            signal, _ = librosa.load(wav_path, sr=SR)

        segs = get_segments_by_label(tsv_path, SR, label)
        if not segs:
            continue

        n_saved = separate_into_folder(signal, segs, type_folder)
        if n_saved:
            done.append(f"{seg_type}={n_saved}")

    if not done:
        return None

    print(f"Done: {stem} — {', '.join(done)}")
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
