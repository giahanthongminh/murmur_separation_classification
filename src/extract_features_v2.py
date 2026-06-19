# extract_features_v2.py
# Unified feature extraction for all three CirCor conditions.
#
# Condition A — Original WAV:    full recording, systole segments only (via TSV)
# Condition B — Separated Murmur: murmur_separated.wav, systole segments only
# Condition C — TSV Systole Murmur (same as B but explicit label for clarity)
#
# All three conditions call the same extract_all_features() from feature_utils
# → identical feature spaces → directly comparable CSVs.
#
# Output:
#   features_v2_original.csv   (~213 features, Condition A)
#   features_v2_separated.csv  (~213 features, Condition B/C)
#
# Usage:
#   python extract_features_v2.py [--condition {all,original,separated}]

import argparse
import os
import pandas as pd
import numpy as np
import librosa
from pathlib import Path

from feature_utils import extract_all_features

DATA_ROOT  = Path("/Users/danggiahan/physionet.org/files/circor-heart-sound/1.0.3")
LABELS_CSV = DATA_ROOT / "labels.csv"
WAV_DIR    = DATA_ROOT / "training_data"   # original WAVs + TSV files
OUTPUT_DIR = DATA_ROOT / "output"          # murmur_separated.wav files
FEAT_DIR   = DATA_ROOT                     # write CSVs here

SR = 4000
LOCATIONS = ["AV", "PV", "TV", "MV"]


def load_systole_signal(signal, sr, tsv_path):
    """Return concatenated systolic segments from TSV, or None if unavailable."""
    if not Path(tsv_path).exists():
        return None

    tsv = pd.read_csv(tsv_path, sep="\t", header=None,
                      names=["start", "end", "label"])
    systole = tsv[tsv["label"] == 2]

    if len(systole) == 0:
        return None

    segments = [
        signal[int(s * sr):int(e * sr)]
        for _, (s, e, _) in systole.iterrows()
        if int(e * sr) > int(s * sr)
    ]
    if not segments:
        return None

    concat = np.concatenate(segments)
    return concat if len(concat) >= 100 else None


def extract_condition_original(labels_df):
    """Condition A: original WAV cropped to systole via TSV."""
    rows = []
    for _, row in labels_df.iterrows():
        pid   = row["Patient ID"]
        label = row["Systolic murmur timing"]

        for loc in LOCATIONS:
            wav_path = WAV_DIR / f"{pid}_{loc}.wav"
            tsv_path = WAV_DIR / f"{pid}_{loc}.tsv"

            if not wav_path.exists():
                continue

            try:
                signal, sr = librosa.load(wav_path, sr=SR)
                systole_sig = load_systole_signal(signal, sr, tsv_path)
                if systole_sig is None:
                    continue

                feats = extract_all_features(systole_sig, sr)
                feats["patient_id"] = pid
                feats["location"]   = loc
                feats["label"]      = label
                rows.append(feats)
            except Exception as e:
                print(f"  [WARN] {pid}_{loc} original: {e}")

    return pd.DataFrame(rows)


def extract_condition_separated(labels_df):
    """Condition B: separated murmur WAV cropped to systole via TSV."""
    rows = []
    for _, row in labels_df.iterrows():
        pid   = row["Patient ID"]
        label = row["Systolic murmur timing"]

        for loc in LOCATIONS:
            wav_path = OUTPUT_DIR / f"{pid}_{loc}" / "murmur_separated.wav"
            tsv_path = WAV_DIR / f"{pid}_{loc}.tsv"

            if not wav_path.exists():
                continue

            try:
                signal, sr = librosa.load(wav_path, sr=SR)
                systole_sig = load_systole_signal(signal, sr, tsv_path)
                if systole_sig is None:
                    continue

                feats = extract_all_features(systole_sig, sr)
                feats["patient_id"] = pid
                feats["location"]   = loc
                feats["label"]      = label
                rows.append(feats)
            except Exception as e:
                print(f"  [WARN] {pid}_{loc} separated: {e}")

    return pd.DataFrame(rows)


def save(df, path):
    path = Path(path)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows → {path}")
    print(df["label"].value_counts().to_string())
    print(f"Feature count: {len([c for c in df.columns if c not in ['patient_id','location','label']])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["all", "original", "separated"],
                        default="all")
    args = parser.parse_args()

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_CSV}")

    labels_df = pd.read_csv(LABELS_CSV)
    print(f"Loaded {len(labels_df)} patients from labels.csv")

    if args.condition in ("all", "original"):
        print("\n=== Condition A: Original WAV (systole segments) ===")
        df_orig = extract_condition_original(labels_df)
        save(df_orig, FEAT_DIR / "features_v2_original.csv")

    if args.condition in ("all", "separated"):
        print("\n=== Condition B: Separated Murmur (systole segments) ===")
        df_sep = extract_condition_separated(labels_df)
        save(df_sep, FEAT_DIR / "features_v2_separated.csv")


if __name__ == "__main__":
    main()
