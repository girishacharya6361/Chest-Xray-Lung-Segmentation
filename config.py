"""Central configuration for the Chest X-Ray Enhancement & Lung Segmentation project."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
LOG_DIR = RESULTS_DIR / "logs"
SPLIT_FILE = RESULTS_DIR / "split.json"
for _d in (IMAGES_DIR, MASKS_DIR, CHECKPOINT_DIR, RESULTS_DIR, CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PROJECT_TITLE = "Chest X-Ray Enhancement and Lung Segmentation Using CNNs and Transformers"

# ---- image / enhancement defaults -------------------------------------------------
IMAGE_SIZE = 256          # training / inference resolution of the deep models
EVAL_SIZE = 256           # resolution at which all methods are compared
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
FREQ_TYPE = "lowpass"     # lowpass | highpass | highboost
FREQ_CUTOFF = 0.30        # fraction of the max frequency radius (0 disables the filter)

# ---- dataset ----------------------------------------------------------------------
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test

# ---- models -----------------------------------------------------------------------
MODELS = {
    "unet": {"label": "CNN: U-Net", "ckpt": CHECKPOINT_DIR / "unet.pt"},
    "hybrid": {"label": "Proposed: CNN-Transformer Hybrid", "ckpt": CHECKPOINT_DIR / "hybrid.pt"},
}
ABLATION_CKPTS = {
    "Complete hybrid model": CHECKPOINT_DIR / "hybrid.pt",
    "Hybrid without CLAHE": CHECKPOINT_DIR / "ablation_no_clahe.pt",
    "Hybrid without Transformer": CHECKPOINT_DIR / "ablation_no_transformer.pt",
}

# Edit with your own team details (shown on the About page).
TEAM = [
    {"name": "Student Name 1", "role": "Team Lead / Deep Learning", "id": "Roll No."},
    {"name": "Student Name 2", "role": "Image Processing", "id": "Roll No."},
    {"name": "Student Name 3", "role": "Frontend / Evaluation", "id": "Roll No."},
]
GUIDE = "Faculty Guide Name"
INSTITUTION = "Your Institution"


def default_model_cfg(kind: str) -> dict:
    cfg = dict(model=kind, size=IMAGE_SIZE, use_clahe=True, clahe_clip=CLAHE_CLIP,
               clahe_tile=CLAHE_TILE, freq_cutoff=0.0, freq_type="lowpass")
    if kind == "unet":
        cfg.update(base_ch=32)
    else:
        cfg.update(backbone="resnet18", embed_dim=256, depth=2, heads=8,
                   use_transformer=True, dropout=0.1, pretrained=False)
    return cfg


def default_train_cfg() -> dict:
    return dict(epochs=60, batch_size=8, lr=1e-3, weight_decay=1e-4, w_dice=0.5, w_bce=0.5,
                patience=15, seed=SEED, num_workers=0)


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def traditional_params() -> dict:
    """Traditional-pipeline parameters: tuned values (results/best_traditional.json) if available."""
    p = dict(clip=CLAHE_CLIP, tile=CLAHE_TILE, cutoff=FREQ_CUTOFF, ftype=FREQ_TYPE)
    best = _load_json(RESULTS_DIR / "best_traditional.json")
    if best:
        p.update({k: best[k] for k in p if k in best})
    return p


def best_config():
    return _load_json(RESULTS_DIR / "best_config.json")
