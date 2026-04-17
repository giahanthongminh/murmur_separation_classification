import pywt
import numpy as np


def dwt_refine(signal, wavelet="db4", level=4):
    """
    Refine signal using wavelet denoising.

    Idea:
    - remove small noisy components
    - keep main structure (normal heart sound)
    """
    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise level from detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Universal threshold
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply soft thresholding to detail coefficients
    new_coeffs = [coeffs[0]]  # keep approximation

    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, threshold, mode="soft"))

    # Reconstruct signal
    refined = pywt.waverec(new_coeffs, wavelet)

    return refined[:len(signal)]