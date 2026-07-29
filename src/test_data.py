import librosa
import matplotlib.pyplot as plt
from config import AUDIO_DIR
from src.data_validation import validate_dataset

wav_dir = AUDIO_DIR
validate_dataset()
files = list(wav_dir.glob("*.wav"))

print("Found:", len(files))

signal, sr = librosa.load(files[0], sr=4000)

print("Loaded:", files[0].name)
print("Shape:", signal.shape)
print("Sample rate:", sr)

plt.plot(signal)
plt.title(files[0].name)
plt.show()
