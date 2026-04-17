from pathlib import Path
import librosa
import matplotlib.pyplot as plt
from ssa import ssa_decompose

wav_dir = Path("/Users/danggiahan/Documents/heart_sounds/wav")
file = list(wav_dir.glob("*.wav"))[0]

print("Loading file...")
signal, sr = librosa.load(file, sr=4000)

# shorten signal for faster SSA
signal = signal[:4000]

print("Running SSA...")
components = ssa_decompose(signal, L=100)

print("Components shape:", components.shape)

# plot first 3 components
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(components[i])
    plt.title(f"Component {i}")

plt.tight_layout()
plt.show()