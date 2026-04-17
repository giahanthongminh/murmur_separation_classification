from pathlib import Path
import librosa
import matplotlib.pyplot as plt
from cssa import cssa_zcr

wav_dir = Path("/Users/danggiahan/Documents/heart_sounds/wav")
file = list(wav_dir.glob("*.wav"))[0]

signal, sr = librosa.load(file, sr=4000)
signal = signal[:4000]

normal, murmur, selected, zcr_values = cssa_zcr(signal, L=100, zcr_threshold=0.05)

print("Selected components:", selected[:10])
print("Number selected:", len(selected))

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(signal)
plt.title("Original")

plt.subplot(3, 1, 2)
plt.plot(normal)
plt.title("Reconstructed Normal (ZCR)")

plt.subplot(3, 1, 3)
plt.plot(murmur)
plt.title("Separated Murmur (ZCR)")
plt.tight_layout()

plt.tight_layout()
plt.show()