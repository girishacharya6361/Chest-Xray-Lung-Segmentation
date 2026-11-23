"""Evaluation metrics: enhancement (PSNR/SSIM), segmentation, performance."""
import os
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---- enhancement ---------------------------------------------------------------------
def psnr(a, b):
    if np.array_equal(a, b):
        return float("inf")
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim(a, b):
    return float(structural_similarity(a, b, data_range=255))


# ---- segmentation --------------------------------------------------------------------
def _surface(m):
    return m & ~ndi.binary_erosion(m)


def hd95(pred, gt):
    """95th-percentile Hausdorff distance in pixels (NaN if one mask is empty)."""
    p, g = pred > 0, gt > 0
    if not p.any() or not g.any():
        return float("nan")
    sp, sg = _surface(p), _surface(g)
    d_to_g = ndi.distance_transform_edt(~sg)
    d_to_p = ndi.distance_transform_edt(~sp)
    d = np.concatenate([d_to_g[sp], d_to_p[sg]])
    return float(np.percentile(d, 95))


def seg_metrics(pred, gt) -> dict:
    p, g = pred > 0, gt > 0
    tp = float((p & g).sum()); fp = float((p & ~g).sum())
    fn = float((~p & g).sum()); tn = float((~p & ~g).sum())
    div = lambda a, b, default=0.0: a / b if b > 0 else default
    dice = div(2 * tp, 2 * tp + fp + fn, 1.0)
    iou = div(tp, tp + fp + fn, 1.0)
    prec = div(tp, tp + fp)
    rec = div(tp, tp + fn)
    f1 = div(2 * prec * rec, prec + rec)
    spec = div(tn, tn + fp)
    return dict(Dice=dice, IoU=iou, Precision=prec, Recall=rec, F1=f1,
                Sensitivity=rec, Specificity=spec, HD95=hd95(pred, gt))


# ---- performance ---------------------------------------------------------------------
def file_size_mb(path) -> float:
    return os.path.getsize(path) / (1024 * 1024) if path and os.path.exists(path) else 0.0
