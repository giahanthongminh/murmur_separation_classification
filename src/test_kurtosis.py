import librosa
import matplotlib.pyplot as plt
from src.cssa import cssa_kurtosis
from config import AUDIO_DIR
from src.data_validation import validate_dataset

wav_dir = AUDIO_DIR
validate_dataset()
file = list(wav_dir.glob("*.wav"))[0]

signal, sr = librosa.load(file, sr=4000)
signal = signal[:4000]

normal, murmur_candidate, selected, kurt_values = cssa_kurtosis(signal, L=100)

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
plt.plot(murmur_candidate)
plt.title("Murmur Candidate (Kurtosis)")

plt.tight_layout()
plt.show()
