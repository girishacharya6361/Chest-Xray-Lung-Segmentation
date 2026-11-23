"""Implementation 2 - U-Net (encoder-decoder CNN with skip connections)."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2. stride>1 downsamples in the first conv."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        c = [base_ch * m for m in (1, 2, 4, 8, 16)]
        self.enc = nn.ModuleList([ConvBlock(in_ch, c[0])] + [ConvBlock(c[i], c[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)                                   # max pooling
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([ConvBlock(c[i] * 2, c[i]) for i in range(4)])
        self.head = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()                                               # bottleneck
        for i in reversed(range(4)):
            x = self.up[i](x)
            skip = skips[i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.dec[i](torch.cat([skip, x], 1))                  # skip connection
        return self.head(x)                                           # logits
