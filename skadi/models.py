"""Network definitions.

* `SkadiUNet`  - the model of the Skadi document (Sec. 3.2): a U-Net whose encoder
  is MobileNetV3-Small and whose decoder is made of depthwise-separable blocks.
* `TinyIceNet` - our re-implementation of the baseline (arXiv:2603.03075, Sec. 4.1),
  trained with exactly the same pipeline so that the comparison does not depend
  only on the number printed in the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .constants import N_SOD


def conv_bn_relu(cin: int, cout: int, k: int = 3, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class DSConv(nn.Sequential):
    """Depthwise 3x3 + pointwise 1x1, each followed by BN and ReLU."""

    def __init__(self, cin: int, cout: int):
        super().__init__(conv_bn_relu(cin, cin, 3, groups=cin), conv_bn_relu(cin, cout, 1))


class UpBlock(nn.Module):
    """Upsample x2, concatenate the encoder skip connection, refine."""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(DSConv(cin + cskip, cout), DSConv(cout, cout))

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


def _adapt_first_conv(conv: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    """Re-use ImageNet filters for a different number of input channels."""
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    with torch.no_grad():
        mean = conv.weight.mean(dim=1, keepdim=True)  # average over R, G, B
        new.weight.copy_(mean.repeat(1, in_ch, 1, 1) * (3.0 / in_ch))
    return new


class SkadiUNet(nn.Module):
    """U-Net with a MobileNetV3-Small encoder.

    Encoder taps (channels @ stride): 16@2, 16@4, 24@8, 48@16, 96@32.
    The classifier works at stride 2 and its logits are bilinearly upsampled to
    the input size: ice charts are drawn as large polygons, so a full-resolution
    decoder stage would only add multiply-accumulates.
    Input height and width must be multiples of 32.
    """

    def __init__(self, in_ch: int = 2, n_classes: int = N_SOD, pretrained: bool = True,
                 widths=(128, 96, 64, 48), dropout: float = 0.1):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        feats = mobilenet_v3_small(weights=weights).features
        feats[0][0] = _adapt_first_conv(feats[0][0], in_ch)
        self.stem = feats[0]  # 16 @ 1/2
        self.enc1 = feats[1]  # 16 @ 1/4
        self.enc2 = feats[2:4]  # 24 @ 1/8
        self.enc3 = feats[4:9]  # 48 @ 1/16
        self.enc4 = feats[9:12]  # 96 @ 1/32

        w4, w3, w2, w1 = widths
        self.bridge = DSConv(96, w4)
        self.up3 = UpBlock(w4, 48, w3)
        self.up2 = UpBlock(w3, 24, w2)
        self.up1 = UpBlock(w2, 16, w1)
        self.up0 = UpBlock(w1, 16, w1)
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(w1, n_classes, 1)
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        s2 = self.stem(x)
        s4 = self.enc1(s2)
        s8 = self.enc2(s4)
        s16 = self.enc3(s8)
        s32 = self.enc4(s16)
        y = self.bridge(s32)
        y = self.up3(y, s16)
        y = self.up2(y, s8)
        y = self.up1(y, s4)
        y = self.up0(y, s2)
        return self.final_up(self.head(self.dropout(y)))


class TinyIceNet(nn.Module):
    """Four double-conv blocks (16, 32, 64, 64 filters), three poolings, x8 upsampling, 1x1 head.

    With 2 input channels and 7 outputs this has 146,599 parameters, matching the
    146.6 k stated in the paper.
    """

    def __init__(self, in_ch: int = 2, n_classes: int = N_SOD):
        super().__init__()

        def dconv(cin, cout):
            return nn.Sequential(conv_bn_relu(cin, cout), conv_bn_relu(cout, cout))

        self.encoder = nn.Sequential(
            dconv(in_ch, 16), nn.MaxPool2d(2),
            dconv(16, 32), nn.MaxPool2d(2),
            dconv(32, 64), nn.MaxPool2d(2),
            dconv(64, 64),
        )
        self.up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        return self.head(self.up(self.encoder(x)))


def build_model(arch: str, in_ch: int = 2, n_classes: int = N_SOD, pretrained: bool = True) -> nn.Module:
    if arch == "skadi":
        return SkadiUNet(in_ch, n_classes, pretrained=pretrained)
    if arch == "tinyicenet":
        return TinyIceNet(in_ch, n_classes)
    raise ValueError(f"unknown architecture '{arch}'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, in_ch: int = 2, size: int = 512) -> float:
    """Multiply-accumulates of all convolution/linear layers for one `size` x `size` input."""
    total = 0

    def hook(mod, _inp, out):
        nonlocal total
        if isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size
            total += out.numel() * kh * kw * mod.in_channels // mod.groups
        elif isinstance(mod, nn.Linear):
            total += out.numel() * mod.in_features

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model(torch.zeros(1, in_ch, size, size, device=device))
    model.train(was_training)
    for h in handles:
        h.remove()
    return float(total)
