from pathlib import Path
import librosa
import matplotlib.pyplot as plt
from cssa import cssa_kurtosis

wav_dir = Path("/Users/danggiahan/Documents/heart_sounds/wav")
file = list(wav_dir.glob("*.wav"))[0]

signal, sr = librosa.load(file, sr=4000)
signal = signal[:4000]

normal, murmur, selected, kurt_values = cssa_kurtosis(signal, L=100, top_k=5)

print("Selected components:", selected)
print("Number selected:", len(selected))

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(signal)
plt.title("Original")

plt.subplot(3, 1, 2)
plt.plot(normal)
plt.title("Reconstructed Normal (Kurtosis)")

plt.subplot(3, 1, 3)
plt.plot(murmur)
plt.title("Separated Murmur (Kurtosis)")

plt.tight_layout()
plt.show()