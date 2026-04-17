from pathlib import Path
import json
import librosa
import soundfile as sf
from cssa import compare_cssa_methods
from dwt_refine import dwt_refine
from multiprocessing import Pool

wav_dir = Path("/Users/danggiahan/Documents/heart_sounds/wav")
output_dir = Path("/Users/danggiahan/Documents/heart_sounds/output")


def process_file(file):
    print("Processing:", file.name)

    # Load audio (10 seconds max)
    signal, sr = librosa.load(file, sr=4000)
    signal = signal[:10 * sr]

    # Step 1: CSSA comparison
    result = compare_cssa_methods(signal, L=100, zcr_threshold=0.05, top_k=5)
    best_normal = result["best_normal"]
    best_method = result["best_method"]

    # Step 2: DWT refinement
    refined_normal = dwt_refine(best_normal)

    # Step 3: final murmur
    final_murmur = signal - refined_normal

    # Save outputs
    record_folder = output_dir / file.stem
    record_folder.mkdir(parents=True, exist_ok=True)

    sf.write(record_folder / "normal_reconstructed.wav", refined_normal, sr)
    sf.write(record_folder / "murmur_separated.wav", final_murmur, sr)

    summary = {
        "file_name": file.name,
        "best_method": best_method,
        "corr_zcr": float(result["corr_zcr"]),
        "corr_kurt": float(result["corr_kurt"]),
        "sample_rate": sr,
        "num_samples": len(signal),
    }

    with open(record_folder / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(wav_dir.glob("*.wav"))
    print(f"Found {len(files)} wav files")

    with Pool(processes=8) as pool:
        summary_rows = pool.map(process_file, files)

    with open(output_dir / "all_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("Done. Outputs saved to:", output_dir)