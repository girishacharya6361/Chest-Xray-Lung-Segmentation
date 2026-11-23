"""Shared training engine used by train_unet.py, train_hybrid.py, tuning and ablation."""
import json
import os
import random
import time
import numpy as np

import config
from models import build_model


def dice_loss(logits, target, eps=1.0):
    import torch
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def make_loss(w_dice, w_bce):
    import torch.nn as nn
    bce = nn.BCEWithLogitsLoss()

    def loss_fn(logits, target):
        return w_dice * dice_loss(logits, target) + w_bce * bce(logits, target)
    return loss_fn


def batch_scores(logits, target):
    import torch
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum((1, 2, 3))
    s_p, s_t = pred.sum((1, 2, 3)), target.sum((1, 2, 3))
    dice = (2 * inter + 1e-6) / (s_p + s_t + 1e-6)
    iou = (inter + 1e-6) / (s_p + s_t - inter + 1e-6)
    return dice.sum().item(), iou.sum().item()


def _write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def run_training(model_cfg, train_cfg, name, save_ckpt=True, log=print):
    """Train, validating every epoch; keep the checkpoint with the best validation Dice.
    History is written to results/history_<name>.json after every epoch (the UI polls it)."""
    import torch
    from training.dataset_loader import get_loaders

    seed = train_cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, split = get_loaders(model_cfg, train_cfg)
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = make_loss(train_cfg["w_dice"], train_cfg["w_bce"])
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    hist_path = config.RESULTS_DIR / f"history_{name}.json"
    ckpt_path = config.CHECKPOINT_DIR / f"{name}.pt"
    hist = dict(name=name, status="running", device=str(device), n_params=n_params,
                epochs_total=train_cfg["epochs"], model_cfg=model_cfg, train_cfg=train_cfg,
                n_train=len(loaders["train"].dataset), n_val=len(loaders["val"].dataset),
                epochs=[], best_val_dice=0.0, best_epoch=0)
    _write_json(hist_path, hist)
    log(f"[{name}] device={device} params={n_params:,} train={hist['n_train']} val={hist['n_val']}")

    best, bad, t_start = -1.0, 0, time.time()
    for ep in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        tr_loss, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0); n += x.size(0)
        tr_loss /= max(n, 1)

        model.eval()
        va_loss, d_sum, i_sum, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(device), y.to(device)
                out = model(x)
                va_loss += loss_fn(out, y).item() * x.size(0)
                d, i = batch_scores(out, y)
                d_sum += d; i_sum += i; n += x.size(0)
        va_loss, va_dice, va_iou = va_loss / n, d_sum / n, i_sum / n
        sched.step(va_dice)

        hist["epochs"].append(dict(epoch=ep, train_loss=tr_loss, val_loss=va_loss, val_dice=va_dice,
                                   val_iou=va_iou, lr=opt.param_groups[0]["lr"], seconds=time.time() - t0))
        if va_dice > best:
            best, bad = va_dice, 0
            hist["best_val_dice"], hist["best_epoch"] = va_dice, ep
            if save_ckpt:
                torch.save(dict(state_dict=model.state_dict(), model_cfg=model_cfg, train_cfg=train_cfg,
                                best_val_dice=va_dice, best_epoch=ep, n_params=n_params), ckpt_path)
        else:
            bad += 1
        _write_json(hist_path, hist)
        log(f"[{name}] epoch {ep:3d}/{train_cfg['epochs']} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_dice={va_dice:.4f} val_iou={va_iou:.4f} best={best:.4f}")
        if bad >= train_cfg["patience"]:
            log(f"[{name}] early stopping (no improvement for {bad} epochs)")
            break

    hist["status"] = "finished"
    hist["train_seconds"] = time.time() - t_start
    _write_json(hist_path, hist)
    log(f"[{name}] finished. best val Dice={best:.4f} at epoch {hist['best_epoch']}")
    return hist


# ---- command-line helpers shared by the two training scripts -----------------------------------
def add_common_args(p):
    d = config.default_train_cfg()
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--w-dice", type=float, default=d["w_dice"])
    p.add_argument("--w-bce", type=float, default=d["w_bce"])
    p.add_argument("--patience", type=int, default=d["patience"])
    p.add_argument("--num-workers", type=int, default=d["num_workers"])
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--clahe-clip", type=float, default=config.CLAHE_CLIP)
    p.add_argument("--no-clahe", action="store_true")
    p.add_argument("--freq-cutoff", type=float, default=0.0, help="0 disables the frequency filter")
    p.add_argument("--name", default=None, help="checkpoint name (default: model kind)")
    p.add_argument("--use-best", action="store_true", help="start from results/best_config.json (tuning output)")
    return p


def cfgs_from_args(a, kind):
    m = config.default_model_cfg(kind)
    m.update(size=a.size, clahe_clip=a.clahe_clip, use_clahe=not a.no_clahe, freq_cutoff=a.freq_cutoff)
    t = config.default_train_cfg()
    t.update(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, w_dice=a.w_dice, w_bce=a.w_bce,
             patience=a.patience, num_workers=a.num_workers, seed=a.seed)
    if a.use_best and config.best_config():
        b = config.best_config()
        shared = ("size", "use_clahe", "clahe_clip", "clahe_tile", "freq_cutoff", "freq_type")
        m.update({k: v for k, v in b["model_cfg"].items() if k in shared or (kind == "hybrid" and k in m)})
        m["model"] = kind
        t.update({k: v for k, v in b["train_cfg"].items() if k in ("batch_size", "lr", "w_dice", "w_bce")})
        print("Using tuned configuration from results/best_config.json")
    return m, t
