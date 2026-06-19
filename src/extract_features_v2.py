# extract_features_v2.py
# Unified feature extraction for all CirCor conditions.
#
# Condition A — Original WAV:      systole segments only (via TSV)
# Condition B — Separated Murmur:  murmur_separated.wav, systole segments only
#
# Key design: per-cycle morphology + onset timing features.
# TSV gives us exact boundaries of each systole cycle → we pass each cycle
# individually to aggregate_per_cycle_features() so timing is not diluted
# by concatenation. Global features (MFCC, PSD, amplitude) still run on the
# full concatenated signal for stable statistics.
#
# Output:
#   features_v2_original.csv   (~242 features, Condition A)
#   features_v2_separated.csv  (~242 features, Condition B)
#
# Usage:
#   python extract_features_v2.py [--condition {all,original,separated}]

import argparse
import pandas as pd
import numpy as np
import librosa
from pathlib import Path

from feature_utils import extract_all_features

DATA_ROOT  = Path("/Users/danggiahan/physionet.org/files/circor-heart-sound/1.0.1")
LABELS_CSV = DATA_ROOT / "labels.csv"
WAV_DIR    = DATA_ROOT / "training_data"
OUTPUT_DIR = DATA_ROOT / "output"
FEAT_DIR   = DATA_ROOT

SR = 4000
LOCATIONS = ["AV", "PV", "TV", "MV"]


def load_systole_cycles(signal, sr, tsv_path):
    """
    Parse TSV and return:
      - cycles      : list of individual systole arrays (per heartbeat cycle)
      - concat      : all cycles concatenated (for global features)
    Returns (None, None) if TSV missing or no systole segments found.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        return None, None

    tsv = pd.read_csv(tsv_path, sep="\t", header=None,
                      names=["start", "end", "label"])
    systole_rows = tsv[tsv["label"] == 2]

    if len(systole_rows) == 0:
        return None, None

    cycles = []
    for _, (s, e, _) in systole_rows.iterrows():
        seg = signal[int(s * sr):int(e * sr)]
        if len(seg) >= 50:
            cycles.append(seg)

    if not cycles:
        return None, None

    concat = np.concatenate(cycles)
    return cycles, concat


def extract_condition_original(labels_df):
    """Condition A: original WAV, systole cycles via TSV."""
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
                cycles, concat = load_systole_cycles(signal, sr, tsv_path)
                if cycles is None:
                    continue

                feats = extract_all_features(concat, sr, cycles=cycles)
                feats["patient_id"] = pid
                feats["location"]   = loc
                feats["label"]      = label
                rows.append(feats)
            except Exception as e:
                print(f"  [WARN] {pid}_{loc} original: {e}")

    return pd.DataFrame(rows)


def extract_condition_separated(labels_df):
    """
    Condition B: separated murmur WAV, systole cycles via TSV.

    murmur_separated.wav was produced by run_all_separation.py which concatenated
    systole segments in TSV order before running CSSA. So the cumulative lengths
    from the TSV map directly into the separated WAV — we recover per-cycle
    boundaries from TSV durations rather than original timestamps.
    """
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

                # Recover per-cycle boundaries from TSV durations
                cycles = _split_by_tsv_durations(signal, sr, tsv_path)
                if cycles is None:
                    continue

                concat = np.concatenate(cycles)

                feats = extract_all_features(concat, sr, cycles=cycles)
                feats["patient_id"] = pid
                feats["location"]   = loc
                feats["label"]      = label
                rows.append(feats)
            except Exception as e:
                print(f"  [WARN] {pid}_{loc} separated: {e}")

    return pd.DataFrame(rows)


def _split_by_tsv_durations(signal, sr, tsv_path):
    """
    Split a concatenated signal back into per-cycle chunks using TSV durations.

    Since murmur_separated.wav = concatenation of systole segments in TSV order,
    each cycle occupies (end - start) * sr samples in the output file.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        return None

    tsv = pd.read_csv(tsv_path, sep="\t", header=None,
                      names=["start", "end", "label"])
    systole_rows = tsv[tsv["label"] == 2]

    if len(systole_rows) == 0:
        return None

    cycles = []
    offset = 0
    for _, (s, e, _) in systole_rows.iterrows():
        n_samples = int((e - s) * sr)
        seg = signal[offset:offset + n_samples]
        if len(seg) >= 50:
            cycles.append(seg)
        offset += n_samples

    return cycles if cycles else None


def save(df, path):
    path = Path(path)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows → {path}")
    print(df["label"].value_counts().to_string())
    n_feats = len([c for c in df.columns if c not in ["patient_id", "location", "label"]])
    print(f"Feature count: {n_feats}")


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
        print("\n=== Condition A: Original WAV (per-cycle systole) ===")
        df_orig = extract_condition_original(labels_df)
        save(df_orig, FEAT_DIR / "features_v2_original.csv")

    if args.condition in ("all", "separated"):
        print("\n=== Condition B: Separated Murmur (per-cycle systole) ===")
        df_sep = extract_condition_separated(labels_df)
        save(df_sep, FEAT_DIR / "features_v2_separated.csv")


if __name__ == "__main__":
    main()
