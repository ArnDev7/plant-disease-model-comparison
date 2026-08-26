# src/models/mlp.py
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Multi-Layer Perceptron (MLP) baseline for plant disease image classification.

    Input:  (B, 3, H, W)   e.g. (B, 3, 224, 224) -> Flattened to (B, 150528)
    Output: (B, num_classes) raw logits

    Architecture:
      Input Flatten -> (B, 3 * H * W)
      Linear(in_features, 512) -> BatchNorm1d(512) -> ReLU -> Dropout(p)
      Linear(512, 256)        -> BatchNorm1d(256) -> ReLU -> Dropout(p)
      Linear(256, 128)        -> BatchNorm1d(128) -> ReLU -> Dropout(p)
      Linear(128, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        in_channels: int = 3,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        in_features = in_channels * img_size * img_size
        self.num_classes = num_classes
        self.in_features = in_features

        layers: List[nn.Module] = [nn.Flatten()]

        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # Final classification head
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
