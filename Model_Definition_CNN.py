from __future__ import annotations

import torch
import torch.nn as nn
import timm
from timm.models.layers import create_conv2d

__all__ = ["ConvNeXt9Ch"]

"""
Module for ConvNeXt9Ch: patched ConvNeXt-Tiny backbone for 9-channel remote sensing.
Loads and modifies pretrained weights for custom remote-sensing channels.
"""

class ConvNeXt9Ch(nn.Module):
    """
    Init a ConvNeXt-Tiny model adapteed for 9-channel inputs.
    Fetches pretrained backbone, patchs the stem convolution, and build the head.
    """
    def __init__(self, num_classes: int, model_name: str = "convnext_tiny"):
        super().__init__()
        # load pretrained backbone without its head
        backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,     # drop the default classifier
            features_only=False
        )

        # patch first convolution: expand in_channels from 3 to 9
        old_conv = backbone.stem[0]
        new_conv = create_conv2d(
            in_channels=9,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        """
        copy RGB weights and init extra channels with the mean RGB weights
        new_conv.weight[:, :3] <- old_conv.weight
        the rest channels <- rgb_mean repeated
        """
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight  # RGB -> channels 0-2
            rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = rgb_mean.repeat(1, 6, 1, 1)
        backbone.stem[0] = new_conv

        # define the classification head
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(backbone.num_features),
            nn.Dropout(0.5),
            nn.Linear(backbone.num_features, backbone.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(backbone.num_features // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass throuh backbone and head to produce raw spcies logits.
        """
        feats = self.backbone(x)  # shape [B, num_features]
        return self.head(feats)
