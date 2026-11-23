"""Performance comparison of the three implementations on the held-out test split.
Run:  python -m evaluation.comparison"""
import argparse
import json
import time
import numpy as np
import pandas as pd

import config
from evaluation.metrics import seg_metrics, psnr, ssim, file_size_mb
from preprocessing.enhancement import apply_clahe
from preprocessing.frequency_filter import frequency_filter
from traditional.segmentation import segment_traditional
from training.dataset_loader import get_split

METRICS = ["Dice", "IoU", "Precision", "Recall", "F1", "Sensitivity", "Specificity", "HD95"]


def traditional_fn():
    p = config.traditional_params()
    return lambda g: segment_traditional(g, p["clip"], p["tile"], p["cutoff"], p["ftype"])


def checkpoint_fn(path):
    from utils.inference import load_model, predict
    lm = load_model(path)
    predict(lm, np.zeros((64, 64), np.uint8))          # warm-up (not timed)
    return (lambda g: predict(lm, g)[0]), lm


def evaluate_method(fn, imgs, masks, stems, label):
    rows = []
    for stem, g, gt in zip(stems, imgs, masks):
        t0 = time.perf_counter()
        pred = fn(g)
        ms = (time.perf_counter() - t0) * 1000
        rows.append(dict(Method=label, Image=stem, **seg_metrics(pred, gt), **{"Time_ms": ms}))
    return rows


def evaluate_checkpoint(path, label, split="test", size=config.EVAL_SIZE):
    """Evaluate one trained checkpoint on a split; returns per-image DataFrame."""
    stems, imgs, masks = get_split(split, size)
    fn, lm = checkpoint_fn(path)
    df = pd.DataFrame(evaluate_method(fn, imgs, masks, stems, label))
    df["Model_MB"] = file_size_mb(path)
    return df


def enhancement_table(imgs):
    p = config.traditional_params()
    rows = []
    for g in imgs:
        c = apply_clahe(g, p["clip"], p["tile"])
        f = frequency_filter(c, p["cutoff"], p["ftype"])
        rows.append(dict(PSNR_CLAHE=psnr(g, c), SSIM_CLAHE=ssim(g, c), PSNR_Filtered=psnr(g, f), SSIM_Filtered=ssim(g, f)))
    df = pd.DataFrame(rows).replace(np.inf, np.nan)
    return pd.DataFrame({"Stage": ["Original -> CLAHE", "Original -> CLAHE + frequency filter"],
                         "PSNR (dB)": [df.PSNR_CLAHE.mean(), df.PSNR_Filtered.mean()],
                         "SSIM": [df.SSIM_CLAHE.mean(), df.SSIM_Filtered.mean()]})


def run_comparison(split="test", size=config.EVAL_SIZE, log=print):
    stems, imgs, masks = get_split(split, size)
    log(f"Evaluating on the '{split}' split: {len(stems)} images at {size}x{size}")
    frames = []
    df = pd.DataFrame(evaluate_method(traditional_fn(), imgs, masks, stems, "Traditional (CLAHE + threshold)"))
    df["Model_MB"] = 0.0
    frames.append(df)
    for kind, label in (("unet", "CNN (U-Net)"), ("hybrid", "Hybrid (CNN-Transformer)")):
        path = config.MODELS[kind]["ckpt"]
        if not path.exists():
            log(f"  skipped {label}: checkpoint {path.name} not found"); continue
        fn, _ = checkpoint_fn(path)
        d = pd.DataFrame(evaluate_method(fn, imgs, masks, stems, label))
        d["Model_MB"] = file_size_mb(path)
        frames.append(d)
    per = pd.concat(frames, ignore_index=True)
    per.to_csv(config.RESULTS_DIR / "comparison_per_image.csv", index=False)
    g = per.groupby("Method", sort=False)
    mean = g[METRICS + ["Time_ms", "Model_MB"]].mean().reset_index()
    std = g[METRICS + ["Time_ms"]].std().reset_index()
    mean.to_csv(config.RESULTS_DIR / "comparison_summary.csv", index=False)
    std.to_csv(config.RESULTS_DIR / "comparison_std.csv", index=False)
    enh = enhancement_table(imgs)
    enh.to_csv(config.RESULTS_DIR / "enhancement_metrics.csv", index=False)
    (config.RESULTS_DIR / "comparison_meta.json").write_text(json.dumps(
        dict(split=split, n_images=len(stems), eval_size=size, traditional_params=config.traditional_params(),
             created=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    _plot(mean)
    log(mean.round(4).to_string(index=False))
    return mean, std, enh, per


def _plot(mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["Dice", "IoU", "Precision", "Recall", "F1"]
    ax = mean.set_index("Method")[cols].plot(kind="bar", figsize=(9, 4.5), color=["#0B2545", "#13A89E", "#8DA9C4", "#5BC0BE", "#3A506B"])
    ax.set_ylabel("score"); ax.set_ylim(0, 1.05); plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "comparison_chart.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=config.EVAL_SIZE)
    a = ap.parse_args()
    run_comparison(a.split, a.size)
