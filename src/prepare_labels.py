# prepare_labels.py
# Read training_data.csv → filter murmur Present cases → save clean labels CSV

import pandas as pd

from config import LABELS_PATH, METADATA_PATH, ensure_output_directories
from src.data_validation import validate_dataset


def main() -> int:
    validate_dataset()
    ensure_output_directories()
    df = pd.read_csv(METADATA_PATH)

# Keep only patients with confirmed murmur
    df = df[df["Murmur"] == "Present"]

# Keep relevant columns only
    df = df[["Patient ID", "Systolic murmur timing"]]

# Drop patients with no systolic timing annotation
    df = df.dropna(subset=["Systolic murmur timing"])

# Drop Late-systolic: only ~3-5 patients in full dataset, too few to classify
    df = df[df["Systolic murmur timing"] != "Late-systolic"]

    df.to_csv(LABELS_PATH, index=False)
    print(f"Total labeled cases: {len(df)}")
    print(df["Systolic murmur timing"].value_counts())
    print(f"Labels: {LABELS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
