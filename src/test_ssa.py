import librosa
import matplotlib.pyplot as plt
from src.ssa import ssa_decompose
from config import AUDIO_DIR
from src.data_validation import validate_dataset

wav_dir = AUDIO_DIR
validate_dataset()
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
