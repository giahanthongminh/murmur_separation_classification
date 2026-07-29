#!/bin/bash
# Setup dependencies for murmur separation + classification pipeline

set -e

echo "=== Installing Python dependencies ==="

python3 -m pip install -r requirements.txt

echo ""
echo "=== Verifying installs ==="
python3 -c "import librosa; print('librosa', librosa.__version__)"
python3 -c "import soundfile; print('soundfile ok')"
python3 -c "import pywt; print('pywt ok')"
python3 -c "import PIL; print('PIL ok')"
python3 -c "import sklearn; print('sklearn', sklearn.__version__)"
python3 -c "import scipy; print('scipy', scipy.__version__)"

echo ""
echo "=== All dependencies ready ==="
echo "For paused classifier experiments, install requirements-classification.txt separately."
