"""Convert the official Montgomery County download into dataset/images and dataset/masks.

Official layout:  MontgomerySet/CXR_png/*.png  and  MontgomerySet/ManualMask/{leftMask,rightMask}/*.png
Usage:  python -m utils.prepare_dataset --src /path/to/MontgomerySet
Left and right lung masks are merged into one binary mask per image (same file name as the image)."""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

import config
from preprocessing.image_loader import read_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MontgomerySet folder")
    a = ap.parse_args()
    src = Path(a.src)
    imgs = sorted((src / "CXR_png").glob("*.png"))
    if not imgs:
        raise SystemExit(f"No PNG images found in {src/'CXR_png'}")
    n = 0
    for ip in imgs:
        left, right = src / "ManualMask" / "leftMask" / ip.name, src / "ManualMask" / "rightMask" / ip.name
        if not (left.exists() and right.exists()):
            print("skip (mask missing):", ip.name); continue
        m = np.maximum(read_mask(left), read_mask(right))
        shutil.copy(ip, config.IMAGES_DIR / ip.name)
        cv2.imwrite(str(config.MASKS_DIR / ip.name), m)
        n += 1
    print(f"Prepared {n} image/mask pairs in {config.DATASET_DIR}")


if __name__ == "__main__":
    main()
