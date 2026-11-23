"""End-to-end processing of one uploaded X-ray: enhancement + three segmentations + metrics."""
import time
import cv2
import numpy as np

import config
from evaluation.metrics import seg_metrics, psnr, ssim
from preprocessing.enhancement import apply_clahe
from preprocessing.frequency_filter import frequency_filter
from preprocessing.image_loader import resize_image
from traditional.segmentation import segment_traditional
from utils import inference

TEAL, NAVY = (19, 168, 158), (11, 37, 69)
METHOD_LABELS = {"traditional": "Traditional", "unet": "CNN (U-Net)", "hybrid": "CNN-Transformer (Proposed)"}


def overlay_mask(gray, mask, color=TEAL, alpha=0.40):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    layer = rgb.copy()
    layer[mask > 0] = color
    out = cv2.addWeighted(layer, alpha, rgb, 1 - alpha, 0)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (255, 190, 0), max(1, gray.shape[0] // 400))
    return out


def enhance(gray, clip, tile, cutoff, ftype, boost=1.0):
    clahe = apply_clahe(gray, clip, tile)
    filt, spec, spec_f, H = frequency_filter(clahe, cutoff, ftype, boost=boost, return_spectra=True)
    return dict(clahe=clahe, filtered=filt, spectrum=spec, spectrum_filtered=spec_f, transfer=H,
                psnr_clahe=psnr(gray, clahe), ssim_clahe=ssim(gray, clahe),
                psnr_filtered=psnr(gray, filt), ssim_filtered=ssim(gray, filt))


def run_pipeline(gray, clip, tile, cutoff, ftype, gt_mask=None):
    res = dict(gray=gray, params=dict(clip=clip, tile=tile, cutoff=cutoff, ftype=ftype), methods={})
    res["enh"] = enhance(gray, clip, tile, cutoff, ftype)
    gt = None
    if gt_mask is not None:
        gt = resize_image(gt_mask, (gray.shape[1], gray.shape[0]), mask=True)
        gt = ((gt > 127) * 255).astype(np.uint8)
    res["gt"] = gt

    t0 = time.perf_counter()
    mask, steps = segment_traditional(gray, clip, tile, cutoff, ftype, return_steps=True)
    res["methods"]["traditional"] = dict(ok=True, mask=mask, ms=(time.perf_counter() - t0) * 1000, steps=steps,
                                         params=f"CLAHE clip={clip}, filter={ftype} cutoff={cutoff}")
    for kind in ("unet", "hybrid"):
        st = inference.model_status(kind)
        if not st["ok"]:
            res["methods"][kind] = dict(ok=False, message=st["message"]); continue
        lm = inference.load_model(st["path"])
        inference.predict(lm, np.zeros((64, 64), np.uint8))          # warm-up so timing is representative
        mask, prob, ms = inference.predict(lm, gray)
        c = lm.cfg
        res["methods"][kind] = dict(ok=True, mask=mask, prob=prob, ms=ms,
                                    params=f"input {c['size']}px, CLAHE={'on' if c.get('use_clahe', True) else 'off'}"
                                           f" (clip {c['clahe_clip']}), val Dice at training {lm.meta.get('best_val_dice') or 0:.3f}")
    for m in res["methods"].values():
        if m["ok"]:
            m["overlay"] = overlay_mask(gray, m["mask"])
            m["metrics"] = seg_metrics(m["mask"], gt) if gt is not None else None
    return res
