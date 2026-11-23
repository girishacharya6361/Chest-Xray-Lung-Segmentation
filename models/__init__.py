def build_model(cfg: dict):
    """Build a model from a config dict (stored inside every checkpoint)."""
    if cfg["model"] == "unet":
        from models.unet import UNet
        return UNet(base_ch=cfg.get("base_ch", 32))
    if cfg["model"] == "hybrid":
        from models.hybrid_model import HybridCNNTransformer
        return HybridCNNTransformer(backbone=cfg.get("backbone", "resnet18"), embed_dim=cfg.get("embed_dim", 256),
                                    depth=cfg.get("depth", 2), heads=cfg.get("heads", 8),
                                    use_transformer=cfg.get("use_transformer", True),
                                    dropout=cfg.get("dropout", 0.1), pretrained=cfg.get("pretrained", False))
    raise ValueError(f"Unknown model type: {cfg['model']}")
