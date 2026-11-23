"""Image / mask loading, matching, normalisation and resizing."""
from pathlib import Path
import cv2
import numpy as np

import config

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def to_uint8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def to_gray(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        a = to_uint8(a[:, :, :3])
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return to_uint8(a)


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes (png/jpg/tif/bmp, 8 or 16 bit, gray or colour) to a uint8 grayscale array."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("Could not decode the file as an image.")
    return to_gray(arr)


def read_gray(path) -> np.ndarray:
    return decode_image(Path(path).read_bytes())


def read_mask(path) -> np.ndarray:
    return ((read_gray(path) > 127) * 255).astype(np.uint8)


def resize_image(img: np.ndarray, size, mask: bool = False) -> np.ndarray:
    """Resize to size x size (or (w,h) tuple). Masks use nearest-neighbour."""
    w, h = (size, size) if isinstance(size, int) else size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    if mask:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if (w * h) < img.shape[0] * img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def normalize(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0


def _list_images(folder: Path):
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMG_EXT)


def find_pairs(images_dir=None, masks_dir=None):
    """Match every image to its mask by file stem (also accepts <stem>_mask / mask_<stem>)."""
    images_dir = Path(images_dir or config.IMAGES_DIR)
    masks_dir = Path(masks_dir or config.MASKS_DIR)
    masks = {p.stem: p for p in _list_images(masks_dir)}
    pairs = []
    for ip in _list_images(images_dir):
        for cand in (ip.stem, ip.stem + "_mask", "mask_" + ip.stem):
            if cand in masks:
                pairs.append((ip.stem, ip, masks[cand]))
                break
    return pairs


def dataset_status() -> dict:
    imgs = _list_images(config.IMAGES_DIR)
    masks = _list_images(config.MASKS_DIR)
    pairs = find_pairs()
    msg = None
    if not imgs:
        msg = "No images found in dataset/images. Download the Montgomery County dataset and run `python -m utils.prepare_dataset --src <MontgomerySet folder>`."
    elif not masks:
        msg = "No masks found in dataset/masks."
    elif not pairs:
        msg = "Images and masks were found but no file names match (image and mask must share the same file name)."
    return dict(n_images=len(imgs), n_masks=len(masks), n_pairs=len(pairs), ready=len(pairs) > 0, message=msg)
