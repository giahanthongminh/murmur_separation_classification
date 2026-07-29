import librosa
import matplotlib.pyplot as plt
from src.cssa import compare_cssa_methods
from config import AUDIO_DIR
from src.data_validation import validate_dataset

wav_dir = AUDIO_DIR
validate_dataset()
file = list(wav_dir.glob("*.wav"))[0]

signal, sr = librosa.load(file, sr=4000)
signal = signal[:4000]

result = compare_cssa_methods(signal, L=100)

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
plt.title(f"Best Murmur Candidate ({result['best_method']})")

plt.tight_layout()
plt.show()
