"""SRGAN-style generator and PatchGAN discriminator for jet calorimeter SR."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.in1(self.conv1(x)))
        h = self.in2(self.conv2(h))
        return x + h


class SRGenerator(nn.Module):
    """
    Upsample LR (B,3,64,64) to HR (B,3,125,125) via bicubic resize + residual refinement.
    Bicubic gives a strong baseline; the CNN learns a residual correction.
    """

    def __init__(self, in_ch: int = 3, feats: int = 64, n_res: int = 8) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, feats, 9, padding=4, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )
        blocks = [ResidualBlock(feats) for _ in range(n_res)]
        self.body = nn.Sequential(*blocks)
        self.conv_mid = nn.Conv2d(feats, feats, 3, padding=1, padding_mode="reflect")
        self.tail = nn.Sequential(
            nn.Conv2d(feats, feats, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(feats, in_ch, 9, padding=4, padding_mode="reflect"),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        # lr: (B,3,64,64) -> coarse HR same size as target
        coarse = F.interpolate(lr, size=(125, 125), mode="bicubic", align_corners=False)
        x = self.head(coarse)
        skip = x
        x = self.body(x)
        x = self.conv_mid(x) + skip
        residual = self.tail(x)
        return coarse + residual


class PatchDiscriminator(nn.Module):
    """PatchGAN with optional class conditioning (projection)."""

    def __init__(self, in_ch: int = 3, feats: int = 64, n_classes: int = 2) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.block1 = nn.Conv2d(in_ch, feats, 4, stride=2, padding=1)
        self.block2 = nn.Conv2d(feats, feats * 2, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(feats * 2, affine=True)
        self.block3 = nn.Conv2d(feats * 2, feats * 4, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(feats * 4, affine=True)
        self.block4 = nn.Conv2d(feats * 4, feats * 8, 4, stride=1, padding=1, bias=False)
        self.bn4 = nn.InstanceNorm2d(feats * 8, affine=True)
        self.out = nn.Conv2d(feats * 8, 1, 4, stride=1, padding=1)
        self.embed = nn.Embedding(n_classes, feats * 8)

    def forward(self, hr: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        h = F.leaky_relu(self.block1(hr), 0.2, inplace=True)
        h = F.leaky_relu(self.bn2(self.block2(h)), 0.2, inplace=True)
        h = F.leaky_relu(self.bn3(self.block3(h)), 0.2, inplace=True)
        feat = F.leaky_relu(self.bn4(self.block4(h)), 0.2, inplace=True)
        if y is not None:
            emb = self.embed(y).view(-1, feat.shape[1], 1, 1)
            feat = feat * (1.0 + torch.tanh(emb))
        return self.out(feat)
