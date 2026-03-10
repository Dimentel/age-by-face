"""
Transformer (ViT) model implementation adapted for age regression.
Based on timm (PyTorch Image Models) vision transformers.
"""

import timm
import torch
from omegaconf import DictConfig
from torch import nn


class AgeTransformer(nn.Module):
    """
    Vision Transformer (ViT) model adapted for age regression.

    Uses timm's implementation of various ViT architectures.
    Supports:
        - Loading pretrained weights from timm
        - Configurable hidden layer (fc parameter) - two linear layers (FC2)
        - Single output for regression
        - Transfer learning mode (custom head)
    """

    def __init__(
        self,
        backbone: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1,
        fc_hidden_size: int = 128,  # size of hidden layer (None for single linear layer)
        output_activation: str | None = "relu",
    ):
        super().__init__()

        # Create backbone using timm
        self.vit = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=1,
        )

        self.output_size = [self.vit.head.in_features]

        # Regression head
        if fc_hidden_size:
            if output_activation:
                self.vit.head = nn.Sequential(
                    nn.LayerNorm(self.output_size[-1]),
                    nn.Linear(self.output_size[-1], fc_hidden_size),
                    nn.Linear(fc_hidden_size, num_classes),
                    nn.ReLU(inplace=True),
                )
            else:
                self.vit.head = nn.Sequential(
                    nn.LayerNorm(self.output_size[-1]),
                    nn.Linear(self.output_size[-1], fc_hidden_size),
                    nn.Linear(fc_hidden_size, num_classes),
                )
        elif output_activation:
            self.vit.head = nn.Sequential(
                nn.LayerNorm(self.output_size[-1]),
                nn.Linear(self.output_size[-1], num_classes),
                nn.ReLU(inplace=True),
            )
        else:
            self.vit.head = nn.Sequential(
                nn.LayerNorm(self.output_size[-1]),
                nn.Linear(self.output_size[-1], num_classes),
            )

        # Initialize head
        self._init_head()

    def _init_head(self):
        """Initialize the regression head weights."""
        for m in self.vit.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "AgeTransformer":
        """Create model from configuration."""
        return cls(
            backbone=cfg.get("backbone", "vit_tiny_patch16_224"),
            pretrained=cfg.get("pretrained", True),
            num_classes=cfg.get("output_dim", 1),
            fc_hidden_size=cfg.get("fc", 128),
            output_activation=cfg.get("output_activation", None),  # null, "relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Age predictions tensor of shape (batch_size, 1)
        """
        return self.vit(x)
