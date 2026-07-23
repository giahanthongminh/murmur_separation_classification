# validate_separation.py
# Dataset-wide validation of CSSA+DWT murmur separation quality.
#
# run_separation_per_segment.py saves only the final murmur array per
# segment — the CSSA correlation scores, which method was chosen, and which
# components were judged "normal heart sound" are computed internally but
# discarded. This script reloads the raw (pre-separation) segment signal for
# every already-processed segment under output_per_seg/, re-runs the CSSA+DWT
# stages to recover those intermediate diagnostics, and persists them
# alongside a sample of visual diagnostics and dataset-wide summary figures.
#
# Key open question this is meant to answer: inside an already-isolated
# systole/diastole segment (no S1/S2 present), does the CSSA "normal"
# component reflect real residual heart-sound structure, or is it just a
# smoother sub-band of the murmur itself? See
# selected_component_energy_center_frac below — if the "normal" component's
# energy sits near the segment edges, that's consistent with S1/S2 boundary
# leakage (expected, fine); if it sits in the middle, CSSA is likely just
# splitting the murmur itself rather than finding real heart-sound structure.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ssa import ssa_decompose
from cssa import compare_cssa_methods, kurtosis_score
from dwt_refine import dwt_refine

DATA_ROOT = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
wav_dir = DATA_ROOT / "training_data"
output_dir = DATA_ROOT / "output_per_seg"
metrics_path = output_dir / "separation_metrics.csv"
diag_dir = output_dir / "diagnostics"

SR = 4000
MIN_SEG_SAMPLES = 100
N_DIAGNOSTIC_SAMPLES = 15
RANDOM_SEED = 42


SEGMENT_TYPES = ["systole", "diastole"]


def segment_folder(patient_id, location, segment_type):
    """Systole lives directly under <patient>_<loc>/ (original flat layout);
    diastole lives in a <patient>_<loc>/diastole/ subfolder."""
    base = output_dir / f"{patient_id}_{location}"
    return base if segment_type == "systole" else base / segment_type


def list_segments():
    """Yield (patient_id, location, segment_type, seg_idx, t_start, t_end,
    duration) for every segment already separated by
    run_separation_per_segment.py, across both systole and diastole."""
    for folder in sorted(output_dir.iterdir()):
        if not folder.is_dir() or folder.name == "diagnostics":
            continue

        patient_id, location = folder.name.rsplit("_", 1)

        for segment_type in SEGMENT_TYPES:
            type_folder = segment_folder(patient_id, location, segment_type)
            meta_path = type_folder / "segments_meta.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            for entry in meta:
                yield (patient_id, location, segment_type, entry["seg_idx"],
                       entry["t_start"], entry["t_end"], entry["duration"])


def load_raw_segment(patient_id, location, t_start, t_end):
    """Reload the original (pre-separation) segment signal from the raw WAV."""
    wav_path = wav_dir / f"{patient_id}_{location}.wav"
    if not wav_path.exists():
        return None
    signal, sr = librosa.load(wav_path, sr=SR)
    s_start, s_end = int(t_start * sr), int(t_end * sr)
    return signal[s_start:s_end]


def rerun_separation(seg_signal):
    """Re-run CSSA+DWT on a raw segment, returning every intermediate stage
    needed for diagnostics (not just the final murmur)."""
    L = min(100, len(seg_signal) // 4)
    result = compare_cssa_methods(seg_signal, L=L, zcr_threshold=0.05)
    components = ssa_decompose(seg_signal, L)

    best_method = result["best_method"]
    selected = (result["selected_zcr"] if best_method == "zcr"
                else result["selected_kurt"])

    refined_normal = dwt_refine(result["best_normal"])[:len(seg_signal)]

    return result, components, selected, refined_normal


def selected_component_energy_center(components, selected, n):
    """Energy-weighted mean time index of the selected 'normal' components,
    normalized to [0, 1]. Near 0/1 = energy near segment edges (consistent
    with S1/S2 boundary leakage). Near 0.5 = energy spread through the
    middle (consistent with CSSA just splitting the murmur itself)."""
    if len(selected) == 0:
        return np.nan

    idx = np.arange(n)
    fracs = []
    for comp_i in selected:
        energy = components[comp_i] ** 2
        total_energy = energy.sum()
        if total_energy <= 0:
            continue
        center = np.sum(energy * idx) / total_energy
        fracs.append(center / max(n - 1, 1))

    return float(np.mean(fracs)) if fracs else np.nan


def compute_segment_metrics(patient_id, location, segment_type, seg_idx, t_start, t_end):
    murmur_path = segment_folder(patient_id, location, segment_type) / f"seg_{seg_idx}_murmur.npy"
    if not murmur_path.exists():
        return None

    seg_signal = load_raw_segment(patient_id, location, t_start, t_end)
    if seg_signal is None or len(seg_signal) < MIN_SEG_SAMPLES:
        return None

    result, components, selected, refined_normal = rerun_separation(seg_signal)

    final_murmur = np.load(murmur_path)
    n = min(len(seg_signal), len(final_murmur))
    seg_signal, final_murmur = seg_signal[:n], final_murmur[:n]

    rms_orig = np.sqrt(np.mean(seg_signal ** 2))
    rms_murmur = np.sqrt(np.mean(final_murmur ** 2))
    energy_ratio_murmur = float(rms_murmur / rms_orig) if rms_orig > 0 else np.nan

    spec_centroid = librosa.feature.spectral_centroid(
        y=final_murmur.astype(np.float32), sr=SR)[0]

    return {
        "corr_zcr": result["corr_zcr"],
        "corr_kurt": result["corr_kurt"],
        "method_chosen": result["best_method"],
        "n_components_total": components.shape[0],
        "n_components_selected_zcr": len(result["selected_zcr"]),
        "n_components_selected_kurt": len(result["selected_kurt"]),
        "energy_ratio_murmur": energy_ratio_murmur,
        "kurtosis_before": kurtosis_score(result["best_normal"]),
        "kurtosis_after": kurtosis_score(refined_normal),
        "spectral_centroid_murmur": float(np.mean(spec_centroid)),
        "selected_component_energy_center_frac": selected_component_energy_center(
            components, selected, len(seg_signal)),
    }


def build_metrics_csv(force=False):
    if metrics_path.exists() and not force:
        print(f"Found existing {metrics_path}, skipping recomputation (use --force to redo).")
        return pd.read_csv(metrics_path)

    rows = []
    for patient_id, location, segment_type, seg_idx, t_start, t_end, duration in list_segments():
        try:
            metrics = compute_segment_metrics(patient_id, location, segment_type,
                                               seg_idx, t_start, t_end)
        except Exception as e:
            print(f"Skipping {patient_id}_{location} {segment_type} seg{seg_idx}: {e}")
            continue
        if metrics is None:
            continue

        row = {"patient_id": patient_id, "location": location, "segment_type": segment_type,
               "seg_idx": seg_idx, "t_start": t_start, "t_end": t_end, "duration": duration}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_path, index=False)
    print(f"Saved {len(df)} segment rows -> {metrics_path}")
    return df


def join_patient_metadata(df):
    """Pull Systolic/Diastolic murmur pitch/grading from the full
    training_data.csv — labels.csv (via prepare_labels.py) only keeps
    Patient ID + systolic timing."""
    raw_path = DATA_ROOT / "training_data.csv"
    if not raw_path.exists():
        print(f"Warning: {raw_path} not found, skipping pitch/grading join.")
        return df

    raw = pd.read_csv(raw_path)
    raw = raw.rename(columns={"Patient ID": "patient_id"})
    raw["patient_id"] = raw["patient_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)

    cols = ["patient_id",
            "Systolic murmur pitch", "Systolic murmur grading",
            "Diastolic murmur pitch", "Diastolic murmur grading"]
    cols = [c for c in cols if c in raw.columns]
    return df.merge(raw[cols], on="patient_id", how="left")


def plot_segment_diagnostics(patient_id, location, segment_type, seg_idx, t_start, t_end):
    murmur_path = segment_folder(patient_id, location, segment_type) / f"seg_{seg_idx}_murmur.npy"
    seg_signal = load_raw_segment(patient_id, location, t_start, t_end)
    if seg_signal is None or not murmur_path.exists() or len(seg_signal) < MIN_SEG_SAMPLES:
        return

    result, _components, _selected, refined_normal = rerun_separation(seg_signal)
    normal = result["best_normal"]
    final_murmur = np.load(murmur_path)
    n = min(len(seg_signal), len(final_murmur))
    seg_signal, final_murmur = seg_signal[:n], final_murmur[:n]

    fig = plt.figure(figsize=(11, 10), constrained_layout=True)
    gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 1.6], figure=fig)

    waveforms = [
        (f"Original segment ({segment_type})", seg_signal),
        (f"CSSA normal ({result['best_method']})", normal),
        ("DWT-refined normal", refined_normal),
        ("Final murmur", final_murmur),
    ]
    for i, (title, sig) in enumerate(waveforms):
        ax = fig.add_subplot(gs[i, :])
        ax.plot(sig)
        ax.set_title(title, fontsize=9)

    def to_db(sig):
        S = np.abs(librosa.stft(np.asarray(sig, dtype=np.float32)))
        return librosa.amplitude_to_db(S, ref=np.max)

    S_orig_db = to_db(seg_signal)
    S_murmur_db = to_db(final_murmur)
    vmin = min(S_orig_db.min(), S_murmur_db.min())
    vmax = max(S_orig_db.max(), S_murmur_db.max())

    ax1 = fig.add_subplot(gs[4, 0])
    librosa.display.specshow(S_orig_db, sr=SR, x_axis="time", y_axis="hz",
                              vmin=vmin, vmax=vmax, ax=ax1)
    ax1.set_title("Spectrogram: original", fontsize=9)

    ax2 = fig.add_subplot(gs[4, 1])
    img = librosa.display.specshow(S_murmur_db, sr=SR, x_axis="time", y_axis="hz",
                                    vmin=vmin, vmax=vmax, ax=ax2)
    ax2.set_title("Spectrogram: final murmur", fontsize=9)
    fig.colorbar(img, ax=[ax1, ax2], format="%+2.0f dB")

    fig.savefig(diag_dir / f"{patient_id}_{location}_{segment_type}_seg{seg_idx}.png", dpi=100)
    plt.close(fig)


def save_diagnostic_samples(df, n=N_DIAGNOSTIC_SAMPLES, seed=RANDOM_SEED):
    """Sample across (patient, segment_type) pairs so both systole and
    diastole get diagnostic coverage, not just whichever comes first."""
    diag_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    pairs = df[["patient_id", "segment_type"]].drop_duplicates().values
    chosen = pairs[rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)]

    for patient_id, segment_type in chosen:
        sub = df[(df["patient_id"] == patient_id) & (df["segment_type"] == segment_type)]
        row = sub.sample(1, random_state=seed).iloc[0]
        plot_segment_diagnostics(row["patient_id"], row["location"], row["segment_type"],
                                  int(row["seg_idx"]), row["t_start"], row["t_end"])

    print(f"Saved {len(chosen)} diagnostic figures -> {diag_dir}")


def plot_correlation_histograms(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["corr_zcr"].dropna(), bins=30, alpha=0.6, label="corr_zcr")
    ax.hist(df["corr_kurt"].dropna(), bins=30, alpha=0.6, label="corr_kurt")
    ax.set_xlabel("correlation between reconstructed-normal and murmur residual")
    ax.set_ylabel("count")

    notes = []
    for segment_type in SEGMENT_TYPES:
        sub = df[df["segment_type"] == segment_type]
        if len(sub) == 0:
            continue
        method_pct = sub["method_chosen"].value_counts(normalize=True) * 100
        notes.append(segment_type + ": " + ", ".join(f"{m} {p:.0f}%" for m, p in method_pct.items()))
    ax.set_title("Correlation scores by method (chosen split — " + "; ".join(notes) + ")")
    ax.legend()
    fig.tight_layout()
    fig.savefig(diag_dir / "hist_correlation.png", dpi=100)
    plt.close(fig)


def plot_energy_ratio_histogram(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for segment_type in SEGMENT_TYPES:
        vals = df.loc[df["segment_type"] == segment_type, "energy_ratio_murmur"].dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=30, alpha=0.6, label=segment_type)
        plotted = True
    ax.set_xlabel("RMS(final murmur) / RMS(original segment)")
    ax.set_ylabel("count")
    ax.set_title("Energy ratio: separated murmur vs. original segment")
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(diag_dir / "hist_energy_ratio.png", dpi=100)
    plt.close(fig)


def plot_component_energy_center_histogram(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for segment_type in SEGMENT_TYPES:
        vals = df.loc[df["segment_type"] == segment_type,
                      "selected_component_energy_center_frac"].dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=30, range=(0, 1), alpha=0.6, label=segment_type)
        plotted = True
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("energy-weighted mean time of 'normal' component (fraction of segment)")
    ax.set_ylabel("count")
    ax.set_title("Where the 'normal' component's energy sits in the segment\n"
                  "(near 0/1 = edge leakage from S1/S2, near 0.5 = no real structure found)")
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(diag_dir / "hist_component_energy_center.png", dpi=100)
    plt.close(fig)


SEGMENT_TYPE_ADJECTIVE = {"systole": "Systolic", "diastole": "Diastolic"}


def plot_pitch_vs_centroid(df, segment_type):
    pitch_col = f"{SEGMENT_TYPE_ADJECTIVE[segment_type]} murmur pitch"
    if pitch_col not in df.columns:
        print(f"Skipping scatter_pitch_vs_centroid_{segment_type}.png: {pitch_col} not joined.")
        return

    sub = df[df["segment_type"] == segment_type]
    order = ["Low", "Medium", "High"]
    present = [p for p in order if (sub[pitch_col] == p).any()]
    groups = [sub.loc[sub[pitch_col] == p, "spectral_centroid_murmur"].dropna() for p in present]
    if not groups:
        print(f"Skipping scatter_pitch_vs_centroid_{segment_type}.png: no pitch labels present.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(groups, tick_labels=present)
    ax.set_ylabel(f"Spectral centroid of separated {segment_type} murmur (Hz)")
    ax.set_xlabel(pitch_col)
    ax.set_title(f"{SEGMENT_TYPE_ADJECTIVE[segment_type]} murmur spectral centroid by annotated pitch")
    fig.tight_layout()
    fig.savefig(diag_dir / f"scatter_pitch_vs_centroid_{segment_type}.png", dpi=100)
    plt.close(fig)


def print_summary(df):
    print(f"\nTotal segments: {len(df)}")
    print(df["segment_type"].value_counts().to_string())

    for segment_type in SEGMENT_TYPES:
        sub = df[df["segment_type"] == segment_type]
        if len(sub) == 0:
            continue
        print(f"\n--- {segment_type} ({len(sub)} segments) ---")

        method_pct = sub["method_chosen"].value_counts(normalize=True) * 100
        print("Method chosen split:")
        for m, p in method_pct.items():
            print(f"  {m}: {p:.1f}%")

        for col in ["corr_zcr", "corr_kurt", "energy_ratio_murmur",
                    "kurtosis_before", "kurtosis_after", "spectral_centroid_murmur",
                    "selected_component_energy_center_frac"]:
            vals = sub[col].dropna()
            if len(vals) == 0:
                continue
            print(f"{col}: mean={vals.mean():.4f}, median={vals.median():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                         help="Recompute separation_metrics.csv even if it already exists.")
    args = parser.parse_args()

    df = build_metrics_csv(force=args.force)
    if len(df) == 0:
        print("No segments found under", output_dir)
        return

    df = join_patient_metadata(df)

    save_diagnostic_samples(df)
    plot_correlation_histograms(df)
    plot_energy_ratio_histogram(df)
    plot_component_energy_center_histogram(df)
    for segment_type in SEGMENT_TYPES:
        plot_pitch_vs_centroid(df, segment_type)

    print_summary(df)


if __name__ == "__main__":
    main()
