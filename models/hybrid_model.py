"""Implementation 3 - proposed CNN-Transformer hybrid segmentation network.

CNN encoder (multi-scale features)  ->  1x1 projection of the deepest features  ->
Transformer encoder (global self-attention over spatial tokens)  ->
feature combination (concat CNN + Transformer features, 1x1 fuse)  ->
U-shaped CNN decoder that uses the CNN skip features  ->  1-channel lung-mask logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import ConvBlock
from models.transformer import TransformerEncoder


class SimpleEncoder(nn.Module):
    """Plain 5-stage CNN encoder (strides 2..32), trained from scratch."""
    channels = [32, 64, 128, 256, 512]

    def __init__(self, in_ch=1):
        super().__init__()
        layers, c = [], in_ch
        for ch in self.channels:
            layers.append(ConvBlock(c, ch, stride=2)); c = ch
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        feats = []
        for l in self.layers:
            x = l(x); feats.append(x)
        return feats


class ResNetEncoder(nn.Module):
    """ResNet-18/34 encoder (1-channel input). Random init unless pretrained=True (needs internet)."""
    channels = [64, 64, 128, 256, 512]

    def __init__(self, name="resnet18", in_ch=1, pretrained=False):
        super().__init__()
        import torchvision
        net = getattr(torchvision.models, name)(weights="DEFAULT" if pretrained else None)
        conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        if pretrained:
            conv1.weight.data = net.conv1.weight.data.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
        self.conv1, self.bn1, self.relu, self.maxpool = conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4

    def forward(self, x):
        f0 = self.relu(self.bn1(self.conv1(x)))                     # 1/2
        f1 = self.layer1(self.maxpool(f0))                          # 1/4
        f2 = self.layer2(f1)                                        # 1/8
        f3 = self.layer3(f2)                                        # 1/16
        f4 = self.layer4(f3)                                        # 1/32
        return [f0, f1, f2, f3, f4]


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = ConvBlock(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


def conv1x1(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class HybridCNNTransformer(nn.Module):
    def __init__(self, backbone="resnet18", embed_dim=256, depth=2, heads=8, use_transformer=True,
                 dropout=0.1, pretrained=False, in_ch=1, stem_ch=32):
        super().__init__()
        self.use_transformer = use_transformer
        self.stem = ConvBlock(in_ch, stem_ch)                       # full-resolution skip
        self.encoder = SimpleEncoder(in_ch) if backbone == "simple" else ResNetEncoder(backbone, in_ch, pretrained)
        ch = self.encoder.channels
        self.proj = conv1x1(ch[4], embed_dim)
        if use_transformer:
            self.transformer = TransformerEncoder(embed_dim, depth, heads, dropout=dropout)
            self.fuse = conv1x1(embed_dim * 2, embed_dim)           # feature combination
        else:                                                       # ablation: CNN-only bottleneck
            self.transformer = None
            self.fuse = ConvBlock(embed_dim, embed_dim)
        d = [256, 128, 64, 32]
        self.up3, self.up2 = Up(embed_dim, ch[3], d[0]), Up(d[0], ch[2], d[1])
        self.up1, self.up0 = Up(d[1], ch[1], d[2]), Up(d[2], ch[0], d[3])
        self.up_full = Up(d[3], stem_ch, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        full = self.stem(x)
        f = self.encoder(x)
        b = self.proj(f[4])
        if self.transformer is not None:
            b = self.fuse(torch.cat([b, self.transformer(b)], 1))
        else:
            b = self.fuse(b)
        d = self.up3(b, f[3]); d = self.up2(d, f[2]); d = self.up1(d, f[1]); d = self.up0(d, f[0])
        return self.head(self.up_full(d, full))                     # logits
