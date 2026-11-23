"""Train the U-Net baseline:  python -m training.train_unet --epochs 60"""
import argparse
from training.trainer import run_training, add_common_args, cfgs_from_args


def train_unet(model_cfg, train_cfg, name="unet", save_ckpt=True, log=print):
    return run_training(model_cfg, train_cfg, name, save_ckpt, log)


if __name__ == "__main__":
    p = add_common_args(argparse.ArgumentParser(description="Train U-Net"))
    p.add_argument("--base-ch", type=int, default=32)
    a = p.parse_args()
    m, t = cfgs_from_args(a, "unet")
    m["base_ch"] = a.base_ch
    train_unet(m, t, a.name or "unet")
