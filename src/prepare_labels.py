'''
Read training_data.csv
→ Filter only murmur Present cases
→ Extract: filename + systolic timing label
→ Save as a clean CSV ready for classification
'''

# prepare_labels.py
# Read training_data.csv and extract labels for classification
# Output: a clean CSV with filename + systolic timing label

import pandas as pd
from pathlib import Path

# Paths
csv_path = Path("/Users/danggiahan/Documents/heart_sounds/training_data.csv")
output_path = Path("/Users/danggiahan/Documents/heart_sounds/labels.csv")

# Load
df = pd.read_csv(csv_path)

# Keep only murmur Present cases
df = df[df["Murmur"] == "Present"]

# Keep only relevant columns
df = df[["Patient ID", "Systolic murmur timing"]]

# Drop rows with no systolic timing
df = df.dropna(subset=["Systolic murmur timing"])

# Drop Late-systolic (only 1 case, too few)
df = df[df["Systolic murmur timing"] != "Late-systolic"]

# After dropping Late-systolic
df["Systolic murmur timing"] = df["Systolic murmur timing"].replace({
    "Early-systolic": "Non-holosystolic",
    "Mid-systolic": "Non-holosystolic"
})
# Save
df.to_csv(output_path, index=False)

print(f"Total labeled cases: {len(df)}")
print(df["Systolic murmur timing"].value_counts())

