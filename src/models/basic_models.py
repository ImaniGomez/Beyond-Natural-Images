# src/models/basic_models.py

from typing import Tuple
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


# ---------- 1. ResNet-18 classifier ----------

def load_resnet18_classifier(device: torch.device) -> nn.Module:
    """
    Load a pretrained ResNet-18 classifier (ImageNet).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.eval()
    model.to(device)
    return model


# ---------- 2. Bicubic x4 Super-Resolution baseline ----------

class BicubicSRx4(nn.Module):
    """
    Simple baseline for super-resolution using bicubic upsampling by factor 4.
    """

    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        )


# ---------- 3. Gaussian blur denoising baseline ----------

class GaussianDenoise(nn.Module):
    """
    Simple Gaussian blur baseline for denoising.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        import numpy as np

        coords = torch.arange(kernel_size) - kernel_size // 2
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(gx**2 + gy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        self.register_buffer("kernel", kernel[None, None, :, :])
        self.kernel_size = kernel_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        kernel = self.kernel.repeat(c, 1, 1, 1)
        padding = self.kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=c)
