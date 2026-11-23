"""Ablation study: complete hybrid vs. no CLAHE vs. no Transformer (same hyper-parameters, same split).
Run:  python -m evaluation.ablation --epochs 60
The complete model is taken from checkpoints/hybrid.pt (train it first with training.train_hybrid)."""
import argparse
import pandas as pd

import config
from evaluation.comparison import evaluate_checkpoint


def run_ablation(epochs=None, retrain=False, log=print):
    from training.train_hybrid import train_hybrid
    full = config.ABLATION_CKPTS["Complete hybrid model"]
    if not full.exists():
        raise FileNotFoundError("checkpoints/hybrid.pt not found. Run `python -m training.train_hybrid` first - "
                                "the ablation compares variants against the complete trained model.")
    from utils.inference import load_model
    ck = load_model(full)
    base_m, base_t = dict(ck.cfg), dict(ck.meta["train_cfg"])
    if epochs:
        base_t["epochs"] = epochs
    variants = {
        "Hybrid without CLAHE": dict(base_m, use_clahe=False),
        "Hybrid without Transformer": dict(base_m, use_transformer=False),
    }
    for label, cfg in variants.items():
        path = config.ABLATION_CKPTS[label]
        if retrain or not path.exists():
            log(f"Training variant: {label}")
            train_hybrid(cfg, base_t, name=path.stem, log=log)
    rows = []
    for label, path in config.ABLATION_CKPTS.items():
        df = evaluate_checkpoint(path, label)
        m = df.drop(columns=["Method", "Image"]).mean()
        rows.append(dict(Variant=label, **m.to_dict()))
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS_DIR / "ablation.csv", index=False)
    log(out.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="epochs for the two ablated variants (default: same as the full model)")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    run_ablation(a.epochs, a.retrain)
