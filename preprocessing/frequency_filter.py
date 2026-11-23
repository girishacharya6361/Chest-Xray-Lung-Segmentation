"""Fourier transform, frequency-domain filtering and inverse Fourier transform."""
import numpy as np


def _radius(shape):
    """Normalised distance from the spectrum centre (0 = DC, 1 = corner)."""
    h, w = shape
    v = (np.arange(h) - h / 2.0) / (h / 2.0)
    u = (np.arange(w) - w / 2.0) / (w / 2.0)
    return np.sqrt(u[None, :] ** 2 + v[:, None] ** 2) / np.sqrt(2.0)


def butterworth_lowpass(shape, cutoff, order=2):
    d = _radius(shape)
    return 1.0 / (1.0 + (d / max(cutoff, 1e-6)) ** (2 * order))


def make_filter(shape, cutoff, ftype="lowpass", order=2, boost=1.0):
    low = butterworth_lowpass(shape, cutoff, order)
    if ftype == "lowpass":
        return low
    if ftype == "highpass":
        return 1.0 - low
    if ftype == "highboost":          # high-frequency emphasis: keeps DC, boosts edges
        return 1.0 + boost * (1.0 - low)
    raise ValueError(f"Unknown filter type: {ftype}")


def fft2(gray):
    """Forward Fourier transform with the zero frequency shifted to the centre."""
    return np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))


def ifft2(F):
    """Inverse Fourier transform of a centred spectrum (real part)."""
    return np.fft.ifft2(np.fft.ifftshift(F)).real


def log_magnitude(F) -> np.ndarray:
    m = np.log1p(np.abs(F))
    lo, hi = m.min(), m.max()
    return ((m - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)


def frequency_filter(gray, cutoff, ftype="lowpass", order=2, boost=1.0, return_spectra=False):
    """FFT -> multiply by transfer function -> inverse FFT. cutoff<=0 disables the filter."""
    if cutoff is None or cutoff <= 0:
        out = gray.copy()
        if return_spectra:
            S = log_magnitude(fft2(gray))
            return out, S, S, np.ones(gray.shape, np.float32)
        return out
    F = fft2(gray)
    H = make_filter(gray.shape, cutoff, ftype, order, boost)
    out = ifft2(F * H)
    if ftype == "highpass":
        lo, hi = out.min(), out.max()
        out = (out - lo) / (hi - lo + 1e-9) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    if return_spectra:
        return out, log_magnitude(F), log_magnitude(F * H), H.astype(np.float32)
    return out
