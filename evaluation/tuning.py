"""Hyper-parameter tuning.
  python -m evaluation.tuning --stage traditional          (CLAHE clip x filter cutoff, no training needed)
  python -m evaluation.tuning --stage dl --epochs 15       (coordinate-descent over the 8 deep-model parameters)
Selection uses the VALIDATION split only (the test split is never touched)."""
import argparse
import copy
import itertools
import json
import time
import numpy as np
import pandas as pd

import config
from evaluation.metrics import seg_metrics

CLIPS = [1.0, 2.0, 3.0, 4.0, 5.0]
CUTOFFS = [0.0, 0.1, 0.2, 0.3, 0.5]
SPACE = {                                   # deep-model search space (one parameter at a time)
    "clahe_clip": [1.0, 2.0, 3.0, 4.0],
    "freq_cutoff": [0.0, 0.3, 0.5],
    "size": [128, 192, 256],
    "backbone": ["simple", "resnet18", "resnet34"],
    "depth": [1, 2, 4],
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "batch_size": [4, 8, 16],
    "loss_weights": [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7), (1.0, 0.0)],
}


def tune_traditional(log=print):
    from training.dataset_loader import get_split
    from traditional.segmentation import segment_traditional
    stems, imgs, masks = get_split("val", config.EVAL_SIZE)
    rows = []
    for clip, cutoff in itertools.product(CLIPS, CUTOFFS):
        d = [seg_metrics(segment_traditional(g, clip, config.CLAHE_TILE, cutoff, "lowpass"), m)["Dice"]
             for g, m in zip(imgs, masks)]
        rows.append(dict(clahe_clip=clip, freq_cutoff=cutoff, val_dice=float(np.mean(d))))
        log(f"clip={clip} cutoff={cutoff} val Dice={rows[-1]['val_dice']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "tuning_traditional.csv", index=False)
    b = df.loc[df.val_dice.idxmax()]
    (config.RESULTS_DIR / "best_traditional.json").write_text(json.dumps(
        dict(clip=float(b.clahe_clip), cutoff=float(b.freq_cutoff), ftype="lowpass", tile=config.CLAHE_TILE,
             val_dice=float(b.val_dice)), indent=1))
    log(f"Best: clip={b.clahe_clip} cutoff={b.freq_cutoff} (val Dice {b.val_dice:.4f})")
    return df


def _apply(m, t, param, value):
    if param == "loss_weights":
        t["w_dice"], t["w_bce"] = value
    elif param in ("clahe_clip", "freq_cutoff", "size", "backbone", "depth"):
        m[param] = value
    else:
        t[param] = value


def tune_dl(epochs=15, params=None, log=print):
    from training.trainer import run_training
    m, t = config.default_model_cfg("hybrid"), config.default_train_cfg()
    t.update(epochs=epochs, patience=max(5, epochs // 3))
    rows, csv = [], config.RESULTS_DIR / "tuning_dl.csv"
    for param in (params or list(SPACE)):
        results = []
        for value in SPACE[param]:
            m2, t2 = copy.deepcopy(m), copy.deepcopy(t)
            _apply(m2, t2, param, value)
            log(f"--- tuning {param} = {value}")
            t0 = time.time()
            h = run_training(m2, t2, f"tune_{param}_{str(value).replace(' ', '')}", save_ckpt=False, log=log)
            rows.append(dict(parameter=param, value=str(value), val_dice=h["best_val_dice"], best_epoch=h["best_epoch"],
                             seconds=time.time() - t0, n_params=h["n_params"]))
            results.append((h["best_val_dice"], value))
            pd.DataFrame(rows).to_csv(csv, index=False)
        best_val = max(results, key=lambda r: r[0])[1]
        _apply(m, t, param, best_val)                       # carry the best value forward
        log(f"==> best {param} = {best_val}")
    t.update(epochs=config.default_train_cfg()["epochs"], patience=config.default_train_cfg()["patience"])
    (config.RESULTS_DIR / "best_config.json").write_text(json.dumps(dict(model_cfg=m, train_cfg=t), indent=1))
    log("Saved results/best_config.json - train the final model with:  python -m training.train_hybrid --use-best")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["traditional", "dl"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--params", nargs="*", choices=list(SPACE), help="subset of parameters to tune")
    a = ap.parse_args()
    tune_traditional() if a.stage == "traditional" else tune_dl(a.epochs, a.params)
