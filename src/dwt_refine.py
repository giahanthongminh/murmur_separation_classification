import numpy as np


def dwt_refine(signal, wavelet="db4", level=4, threshold_method="soft"):
    """
    Refine signal using wavelet denoising.

    Idea:
    - remove small noisy components
    - keep main structure (normal heart sound)
    """
    try:
        import pywt
    except ImportError as exc:
        raise RuntimeError(
            "DWT refinement requires PyWavelets; install project requirements"
        ) from exc

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("DWT refinement expects a one-dimensional signal")
    if threshold_method not in {"soft", "hard"}:
        raise ValueError("threshold_method must be 'soft' or 'hard'")
    wavelet_object = pywt.Wavelet(wavelet)
    safe_level = min(level, pywt.dwt_max_level(len(values), wavelet_object.dec_len))
    coeffs = pywt.wavedec(values, wavelet_object, level=safe_level)

    # Estimate noise level from detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Universal threshold
    threshold = sigma * np.sqrt(2 * np.log(len(values)))

    # Apply soft thresholding to detail coefficients
    new_coeffs = [coeffs[0]]  # keep approximation

    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, threshold, mode=threshold_method))

    # Reconstruct signal
    refined = pywt.waverec(new_coeffs, wavelet)

    return refined[:len(values)]
