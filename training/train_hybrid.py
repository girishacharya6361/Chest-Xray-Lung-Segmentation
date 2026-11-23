"""Train the CNN-Transformer hybrid:  python -m training.train_hybrid --epochs 60"""
import argparse
from training.trainer import run_training, add_common_args, cfgs_from_args


def train_hybrid(model_cfg, train_cfg, name="hybrid", save_ckpt=True, log=print):
    return run_training(model_cfg, train_cfg, name, save_ckpt, log)


if __name__ == "__main__":
    p = add_common_args(argparse.ArgumentParser(description="Train CNN-Transformer hybrid"))
    p.add_argument("--backbone", default="resnet18", choices=["simple", "resnet18", "resnet34"])
    p.add_argument("--depth", type=int, default=2, help="number of Transformer blocks")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--no-transformer", action="store_true", help="ablation: CNN-only bottleneck")
    p.add_argument("--pretrained", action="store_true", help="ImageNet weights (needs internet)")
    a = p.parse_args()
    m, t = cfgs_from_args(a, "hybrid")
    if not a.use_best:
        m.update(backbone=a.backbone, depth=a.depth)
    m.update(embed_dim=a.embed_dim, heads=a.heads, use_transformer=not a.no_transformer, pretrained=a.pretrained)
    train_hybrid(m, t, a.name or "hybrid")
