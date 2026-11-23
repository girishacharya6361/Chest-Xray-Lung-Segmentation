"""Dataset loading, caching, train/val/test splitting and PyTorch loaders."""
import json
import random
import numpy as np

import config
from preprocessing.image_loader import find_pairs, read_gray, read_mask, resize_image
from preprocessing.enhancement import preprocess_for_model


class DatasetMissing(RuntimeError):
    pass


def load_arrays(size):
    """Load every image/mask pair resized to size x size (cached in results/cache)."""
    pairs = find_pairs()
    if not pairs:
        raise DatasetMissing("No image/mask pairs found in dataset/images and dataset/masks.")
    stems = [p[0] for p in pairs]
    cache = config.CACHE_DIR / f"arrays_{size}.npz"
    if cache.exists():
        z = np.load(cache)
        if z["stems"].tolist() == stems:
            return stems, z["imgs"], z["masks"]
    imgs, masks = [], []
    for stem, ip, mp in pairs:
        img, m = read_gray(ip), read_mask(mp)
        if m.shape != img.shape:
            m = resize_image(m, (img.shape[1], img.shape[0]), mask=True)
        imgs.append(resize_image(img, size))
        masks.append(((resize_image(m, size, mask=True) > 127) * 255).astype(np.uint8))
    imgs, masks = np.stack(imgs), np.stack(masks)
    np.savez_compressed(cache, stems=np.array(stems), imgs=imgs, masks=masks)
    return stems, imgs, masks


def make_splits(stems, seed=config.SEED, ratios=config.SPLIT_RATIOS):
    """Deterministic train/val/test split, stored in results/split.json so all models share it."""
    if config.SPLIT_FILE.exists():
        d = json.loads(config.SPLIT_FILE.read_text())
        if sorted(d["train"] + d["val"] + d["test"]) == sorted(stems):
            return d
    s = sorted(stems)
    random.Random(seed).shuffle(s)
    n_tr, n_va = int(round(len(s) * ratios[0])), int(round(len(s) * ratios[1]))
    d = dict(train=s[:n_tr], val=s[n_tr:n_tr + n_va], test=s[n_tr + n_va:], seed=seed, ratios=list(ratios))
    config.SPLIT_FILE.write_text(json.dumps(d, indent=1))
    return d


def get_split(split, size):
    stems, imgs, masks = load_arrays(size)
    idx = [stems.index(s) for s in make_splits(stems)[split]]
    return [stems[i] for i in idx], imgs[idx], masks[idx]


class XrayDataset:
    def __init__(self, imgs, masks, cfg, augment=False):
        import torch
        self.torch = torch
        self.imgs, self.masks, self.cfg, self.augment = imgs, masks, cfg, augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from training.augmentation import augment_pair
        img, m, c = self.imgs[i], self.masks[i], self.cfg
        if self.augment:
            img, m = augment_pair(img, m)
        x = preprocess_for_model(img, c["size"], c.get("use_clahe", True), c["clahe_clip"],
                                 c.get("clahe_tile", 8), c.get("freq_cutoff", 0.0), c.get("freq_type", "lowpass"))
        y = (m > 127).astype(np.float32)
        return self.torch.from_numpy(x)[None], self.torch.from_numpy(y)[None]


def get_loaders(model_cfg, train_cfg):
    import torch
    from torch.utils.data import DataLoader
    stems, imgs, masks = load_arrays(model_cfg["size"])
    sp = make_splits(stems)
    out = {}
    for name in ("train", "val", "test"):
        idx = [stems.index(s) for s in sp[name]]
        ds_ = XrayDataset(imgs[idx], masks[idx], model_cfg, augment=(name == "train"))
        bs = train_cfg["batch_size"]
        out[name] = DataLoader(ds_, batch_size=bs, shuffle=(name == "train"),
                               drop_last=(name == "train" and len(ds_) > bs),
                               num_workers=train_cfg.get("num_workers", 0))
    return out, sp
