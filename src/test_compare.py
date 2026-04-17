from pathlib import Path
import librosa
import matplotlib.pyplot as plt
from cssa import compare_cssa_methods

wav_dir = Path("/Users/danggiahan/Documents/heart_sounds/wav")
file = list(wav_dir.glob("*.wav"))[0]

signal, sr = librosa.load(file, sr=4000)
signal = signal[:4000]

result = compare_cssa_methods(signal, L=100, zcr_threshold=0.05, top_k=5)

print("Best method:", result["best_method"])
print("corr_zcr:", result["corr_zcr"])
print("corr_kurt:", result["corr_kurt"])

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(signal)
plt.title("Original")

plt.subplot(3, 1, 2)
plt.plot(result["best_normal"])
plt.title(f"Best Normal ({result['best_method']})")

plt.subplot(3, 1, 3)
plt.plot(result["best_murmur"])
plt.title(f"Best Murmur ({result['best_method']})")

plt.tight_layout()
plt.show()