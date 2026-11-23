"""Implementation 1 - traditional lung segmentation.
CLAHE -> (frequency low-pass) -> Gaussian blur -> Otsu threshold -> morphology ->
connected-component analysis -> contour filling."""
import cv2
import numpy as np

from preprocessing.enhancement import apply_clahe
from preprocessing.frequency_filter import frequency_filter


def _odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def segment_traditional(gray, clip=2.0, tile=8, cutoff=0.0, ftype="lowpass",
                        max_side=512, return_steps=False):
    """Return a binary lung mask (uint8, 0/255) at the input resolution."""
    h0, w0 = gray.shape
    s = min(1.0, max_side / max(h0, w0))
    g = cv2.resize(gray, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    h, w = g.shape
    m = max(h, w)
    steps = {}

    enh = apply_clahe(g, clip, tile)                       # 1. contrast enhancement
    steps["CLAHE"] = enh
    if cutoff and cutoff > 0:                               # 2. frequency-domain smoothing
        enh = frequency_filter(enh, cutoff, ftype)
        steps["Frequency filter"] = enh
    blur = cv2.GaussianBlur(enh, (0, 0), sigmaX=max(1.0, 0.004 * m))

    # 3. Otsu threshold computed on the central region (ignores black scanner borders)
    my, mx = int(0.1 * h), int(0.1 * w)
    core = blur[my:h - my, mx:w - mx]
    t, _ = cv2.threshold(core, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (blur < t).astype(np.uint8)                    # lungs are the dark (air) regions
    steps["Threshold (Otsu)"] = binary * 255

    # 4. morphological opening / closing
    k = _odd(max(3, 0.012 * m))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, ker2)
    steps["Morphology"] = morph * 255

    # 5. connected components: drop background touching the frame, keep the two largest regions
    n, lab, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
    frame = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    touch = np.bincount(frame, minlength=n)
    cands = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= 0.01 * h * w]
    good = [c for c in cands if touch[c[1]] <= 0.2 * m]
    pick = sorted(good or cands, reverse=True)[:2]
    keep = np.zeros((h, w), np.uint8)
    for _, i in pick:
        keep[lab == i] = 1
    steps["Connected components"] = keep * 255

    # 6. contour detection + hole filling + smoothing
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    filled = cv2.medianBlur(filled * 255, _odd(max(3, 0.01 * m)))
    mask = ((filled > 127) * 255).astype(np.uint8)
    steps["Final mask"] = mask

    if (w, h) != (w0, h0):
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return (mask, steps) if return_steps else mask
