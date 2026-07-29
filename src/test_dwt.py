import librosa
import matplotlib.pyplot as plt
from src.cssa import compare_cssa_methods
from src.dwt_refine import dwt_refine
from config import AUDIO_DIR
from src.data_validation import validate_dataset

wav_dir = AUDIO_DIR
validate_dataset()
file = list(wav_dir.glob("*.wav"))[0]

signal, sr = librosa.load(file, sr=4000)
signal = signal[:4000]

# Step 1: CSSA (best method)
result = compare_cssa_methods(signal, L=100)

normal = result["best_normal"]

# Step 2: DWT refinement
refined_normal = dwt_refine(normal)

# Step 3: final murmur
murmur_candidate = signal - refined_normal

print("Best method:", result["best_method"])

plt.figure(figsize=(10, 6))

plt.subplot(4, 1, 1)
plt.plot(signal)
plt.title("Original")

plt.subplot(4, 1, 2)
plt.plot(normal)
plt.title("Normal (CSSA)")

plt.subplot(4, 1, 3)
plt.plot(refined_normal)
plt.title("Normal (after DWT)")

plt.subplot(4, 1, 4)
plt.plot(murmur_candidate)
plt.title("Murmur Candidate")

plt.tight_layout()
plt.show()
