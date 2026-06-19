# run_all_separation.py
# For each patient in labels.csv:
#   1. Load WAV and crop to systolic segments (using TSV)
#   2. Run CSSA to separate normal heart sound from murmur
#   3. Refine with DWT
#   4. Save murmur_separated.wav + normal_reconstructed.wav

from pathlib import Path
import json
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from cssa import compare_cssa_methods
from dwt_refine import dwt_refine
from multiprocessing import Pool

DATA_ROOT = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
wav_dir   = DATA_ROOT / "training_data"
tsv_dir   = DATA_ROOT / "training_data"   # TSV files live alongside WAVs
output_dir = DATA_ROOT / "output"
labels_path = DATA_ROOT / "labels.csv"


def get_systolic_segments(tsv_path, sr):
    """Return list of (start_sample, end_sample) for systolic periods (label=2)."""
    tsv = pd.read_csv(tsv_path, sep="\t", header=None, names=["start", "end", "label"])
    segments = tsv[tsv["label"] == 2]
    return [(int(r["start"] * sr), int(r["end"] * sr)) for _, r in segments.iterrows()]


def process_file(file):
    out_folder = output_dir / file.stem

    # Skip if already processed (resumability)
    if (out_folder / "murmur_separated.wav").exists():
        return None

    tsv_path = tsv_dir / (file.stem + ".tsv")
    signal, sr = librosa.load(file, sr=4000)

    # Pre-crop to systolic segments before CSSA
    # Reduces signal length ~60-70% → SVD ~9x faster
    if tsv_path.exists():
        segs = get_systolic_segments(tsv_path, sr)
        if segs:
            signal = np.concatenate([signal[s:e] for s, e in segs])

    if len(signal) < 100:
        return None

    # Stage 1: CSSA — compare ZCR and kurtosis methods, keep lower correlation
    result = compare_cssa_methods(signal, L=100, zcr_threshold=0.05, top_k=5)
    best_normal = result["best_normal"]

    # Stage 2: DWT refinement — remove residual murmur from normal sound
    refined_normal = dwt_refine(best_normal)

    # Final murmur = original - refined normal
    final_murmur = signal - refined_normal

    # Save outputs
    out_folder.mkdir(parents=True, exist_ok=True)
    sf.write(out_folder / "normal_reconstructed.wav", refined_normal, sr)
    sf.write(out_folder / "murmur_separated.wav", final_murmur, sr)

    summary = {
        "file_name":  file.name,
        "best_method": result["best_method"],
        "corr_zcr":   float(result["corr_zcr"]),
        "corr_kurt":  float(result["corr_kurt"]),
    }
    with open(out_folder / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done: {file.name}")
    return summary


if __name__ == "__main__":
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only process patients that have labels (skip the other ~2600 files)
    labels = pd.read_csv(labels_path)
    patient_ids = set(labels["Patient ID"].astype(str))
    files = [f for f in wav_dir.glob("*.wav")
             if f.stem.split("_")[0] in patient_ids]

    print(f"Processing {len(files)} files")

    with Pool(processes=8) as pool:
        pool.map(process_file, files)

    print("Done. Outputs saved to:", output_dir)