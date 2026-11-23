"""Load trained checkpoints and predict lung masks for new images."""
import time
from pathlib import Path
import cv2
import numpy as np

import config
from preprocessing.enhancement import preprocess_for_model

_CACHE = {}


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


class LoadedModel:
    def __init__(self, model, cfg, device, path, meta):
        self.model, self.cfg, self.device, self.path, self.meta = model, cfg, device, Path(path), meta


def load_model(path) -> LoadedModel:
    import torch
    from models import build_model
    path = Path(path)
    key = (str(path), path.stat().st_mtime)
    if key in _CACHE:
        return _CACHE[key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    meta = {k: ck.get(k) for k in ("best_val_dice", "best_epoch", "n_params", "train_cfg")}
    _CACHE[key] = LoadedModel(model, ck["model_cfg"], device, path, meta)
    return _CACHE[key]


def model_status(kind, path=None) -> dict:
    """Why a model can / cannot be used right now."""
    path = Path(path or config.MODELS[kind]["ckpt"])
    if not torch_available():
        return dict(ok=False, path=path, message="PyTorch is not installed (pip install torch torchvision).")
    if not path.exists():
        return dict(ok=False, path=path,
                    message=f"Trained checkpoint not found ({path.name}). Train it first: python -m training.train_{kind} (or use the Model Training page).")
    return dict(ok=True, path=path, message="Ready")


def predict(lm: LoadedModel, gray: np.ndarray, threshold: float = 0.5):
    """Return (binary mask uint8 0/255 at input resolution, probability map, milliseconds)."""
    import torch
    c = lm.cfg
    h, w = gray.shape
    t0 = time.perf_counter()
    x = preprocess_for_model(gray, c["size"], c.get("use_clahe", True), c["clahe_clip"],
                             c.get("clahe_tile", 8), c.get("freq_cutoff", 0.0), c.get("freq_type", "lowpass"))
    xt = torch.from_numpy(x)[None, None].to(lm.device)
    with torch.no_grad():
        prob = torch.sigmoid(lm.model(xt))[0, 0].float().cpu().numpy()   # .cpu() synchronises GPU
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    ms = (time.perf_counter() - t0) * 1000.0
    return ((prob >= threshold) * 255).astype(np.uint8), prob, ms
