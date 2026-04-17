"""U-Net segmentation model for patch-based deforestation detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Simple U-Net for binary segmentation."""

    def __init__(self, in_channels: int = 32, base_filters: int = 64, depth: int = 4):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch * 2
        for i in range(depth - 1, -1, -1):
            out_ch = base_filters * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.decoders.append(ConvBlock(out_ch * 2, out_ch))
            ch = out_ch

        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Handle size mismatch from non-power-of-2 inputs
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)  # raw logits, apply sigmoid externally
