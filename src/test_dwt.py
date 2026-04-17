from pathlib import Path
import librosa
import matplotlib.pyplot as plt
from cssa import compare_cssa_methods
from dwt_refine import dwt_refine

wav_dir = Path("/Users/danggiahan/Documents/heart_sounds/wav")
file = list(wav_dir.glob("*.wav"))[0]

signal, sr = librosa.load(file, sr=4000)
signal = signal[:4000]

# Step 1: CSSA (best method)
result = compare_cssa_methods(signal, L=100)

normal = result["best_normal"]

# Step 2: DWT refinement
refined_normal = dwt_refine(normal)

# Step 3: final murmur
final_murmur = signal - refined_normal

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
plt.plot(final_murmur)
plt.title("Final Murmur")

plt.tight_layout()
plt.show()