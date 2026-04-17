from pathlib import Path
import librosa
import matplotlib.pyplot as plt

wav_dir = Path("/Users/danggiahan/Documents/heart_sounds/wav")
files = list(wav_dir.glob("*.wav"))

print("Found:", len(files))

signal, sr = librosa.load(files[0], sr=4000)

print("Loaded:", files[0].name)
print("Shape:", signal.shape)
print("Sample rate:", sr)

plt.plot(signal)
plt.title(files[0].name)
plt.show()