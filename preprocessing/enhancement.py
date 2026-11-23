"""CLAHE enhancement and the shared model-input preprocessing."""
import cv2
import numpy as np

from preprocessing.frequency_filter import frequency_filter
from preprocessing.image_loader import resize_image, normalize


def apply_clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return clahe.apply(gray)


def preprocess_for_model(gray, size, use_clahe=True, clip=2.0, tile=8, cutoff=0.0, ftype="lowpass"):
    """resize -> (CLAHE) -> (frequency filter) -> [0,1] float32.
    Used identically for training and inference."""
    img = resize_image(gray, int(size))
    if use_clahe:
        img = apply_clahe(img, clip, tile)
    if cutoff and cutoff > 0:
        img = frequency_filter(img, cutoff, ftype)
    return normalize(img)
