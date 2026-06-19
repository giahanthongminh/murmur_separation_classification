# prepare_labels.py
# Read training_data.csv → filter murmur Present cases → save clean labels CSV

import pandas as pd
from pathlib import Path

DATA_ROOT = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
csv_path = DATA_ROOT / "training_data.csv"
output_path = DATA_ROOT / "labels.csv"

df = pd.read_csv(csv_path)

# Keep only patients with confirmed murmur
df = df[df["Murmur"] == "Present"]

# Keep relevant columns only
df = df[["Patient ID", "Systolic murmur timing"]]

# Drop patients with no systolic timing annotation
df = df.dropna(subset=["Systolic murmur timing"])

# Drop Late-systolic: only ~3-5 patients in full dataset, too few to classify
df = df[df["Systolic murmur timing"] != "Late-systolic"]

df.to_csv(output_path, index=False)
print(f"Total labeled cases: {len(df)}")
print(df["Systolic murmur timing"].value_counts())