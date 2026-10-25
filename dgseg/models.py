"""Shared DeepLabV3+ backbone with source-only DG training variants."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


class MixStyle(nn.Module):
    """ICLR 2021 feature-statistics mixing, applied to shallow ResNet features."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or x.size(0) < 2 or torch.rand(()) >= self.p:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True).detach()
        std = (x.var(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6).sqrt().detach()
        normalized = (x - mean) / std
        permutation = torch.randperm(x.size(0), device=x.device)
        mix = torch.distributions.Beta(self.alpha, self.alpha).sample((x.size(0), 1, 1, 1)).to(x.device)
        mixed_mean = mix * mean + (1 - mix) * mean[permutation]
        mixed_std = mix * std + (1 - mix) * std[permutation]
        return normalized * mixed_std + mixed_mean


class RandConv(nn.Module):
    """Segmentation adaptation of ICLR 2021 random-convolution augmentation."""

    def __init__(self, p: float = 0.5, kernels: tuple[int, ...] = (1, 3, 5, 7)):
        super().__init__()
        self.p = p
        self.kernels = kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(()) >= self.p:
            return x
        batch, channels, height, width = x.shape
        kernel = self.kernels[int(torch.randint(len(self.kernels), (1,)).item())]
        weights = torch.randn(batch * channels, channels, kernel, kernel, device=x.device, dtype=x.dtype)
        weights /= (channels * kernel * kernel) ** 0.5
        out = F.conv2d(x.reshape(1, batch * channels, height, width), weights, padding=kernel // 2, groups=batch)
        out = out.reshape(batch, channels, height, width)
        original_mean = x.mean(dim=(2, 3), keepdim=True)
        original_std = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        out = (out - out.mean(dim=(2, 3), keepdim=True)) / out.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        out = out * original_std + original_mean
        alpha = torch.rand(batch, 1, 1, 1, device=x.device, dtype=x.dtype)
        return alpha * x + (1 - alpha) * out


def _conv_bn_relu(in_channels: int, out_channels: int, kernel: int, dilation: int = 1) -> nn.Sequential:
    padding = (kernel // 2) * dilation
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.branches = nn.ModuleList([_conv_bn_relu(in_channels, out_channels, 1)] + [_conv_bn_relu(in_channels, out_channels, 3, rate) for rate in (6, 12, 18)])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), _conv_bn_relu(in_channels, out_channels, 1))
        self.project = _conv_bn_relu(out_channels * 5, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([branch(x) for branch in self.branches] + [pooled], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int, method: str = "erm", pretrained_backbone: bool = False):
        super().__init__()
        if method not in ("erm", "mixstyle", "randconv"):
            raise ValueError(f"Unsupported method: {method}")
        self.method = method
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        self.backbone = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        self.mixstyle = MixStyle() if method == "mixstyle" else nn.Identity()
        self.randconv = RandConv() if method == "randconv" else nn.Identity()
        self.aspp = ASPP(2048)
        self.low_projection = _conv_bn_relu(256, 48, 1)
        self.decoder = nn.Sequential(_conv_bn_relu(304, 256, 3), _conv_bn_relu(256, 256, 3), nn.Conv2d(256, num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        x = self.randconv(x)
        net = self.backbone
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        low = net.layer1(x)
        x = self.mixstyle(low)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.layer4(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat((x, self.low_projection(low)), dim=1))
        return F.interpolate(x, size=original_size, mode="bilinear", align_corners=False)
