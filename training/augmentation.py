"""Joint image/mask augmentation (no horizontal flip: heart position is anatomically meaningful)."""
import cv2
import numpy as np


def augment_pair(img, mask, rng=None):
    rng = rng or np.random.default_rng()
    h, w = img.shape
    ang = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.1)
    tx, ty = rng.uniform(-0.05, 0.05, 2) * (w, h)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    f = img.astype(np.float32) / 255.0
    f = np.power(f, rng.uniform(0.8, 1.25))                         # gamma
    f = (f - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.08, 0.08)   # contrast / brightness
    if rng.random() < 0.3:
        f = f + rng.normal(0, 0.015, f.shape)                       # gaussian noise
    return (np.clip(f, 0, 1) * 255).astype(np.uint8), mask
