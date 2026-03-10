"""
Hybrid ConvNeXT + Transformer model for age regression.
Based on the implementation from:
"Integrating ConvNeXt and vision transformers for enhancing facial age estimation"
"""

import torch
from omegaconf import DictConfig
from torch import nn

from age_by_face.models.convnext import AgeConvNeXt
from age_by_face.models.transformer import AgeTransformer


class ConvNeXTTransformer(nn.Module):
    """
    Hybrid model combining ConvNeXt and Vision Transformer.

    Architecture:
        1. ConvNeXt backbone extracts features from image
        2. Features are reshaped to sequence for Transformer
        3. Transformer processes the sequence
        4. Regression head produces age prediction

    Supports loading pretrained weights from individual checkpoints
    for both ConvNeXt and Transformer parts.
    """

    def __init__(
        self,
        convnext_cfg: DictConfig,
        transformer_cfg: DictConfig,
    ):
        super().__init__()

        # Create ConvNeXt part
        self.ConvNeXt = AgeConvNeXt.from_cfg(convnext_cfg)

        # Create Transformer part
        self.Transformer = AgeTransformer.from_cfg(transformer_cfg)
        self.Transformer.vit.patch_embed = nn.Identity()

    def load_convnext_weights(self, checkpoint_path: str):
        """Load pretrained ConvNeXt weights."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Load ConvNeXt part
        fixed_weights = {}
        for k, v in state_dict.items():
            fixed_key = k.removeprefix("model.")  # delete 'model.'
            fixed_weights[fixed_key] = v

        self.ConvNeXt.load_state_dict(fixed_weights, strict=False)
        print(f"Loaded ConvNeXt weights from {checkpoint_path}")

    def load_transformer_weights(self, checkpoint_path: str):
        """Load pretrained Transformer weights."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Load Transformer part
        fixed_weights = {}
        for k, v in state_dict.items():
            fixed_key = k.removeprefix("model.")  # delete 'model.'
            fixed_weights[fixed_key] = v

        self.Transformer.load_state_dict(fixed_weights, strict=False)
        print(f"Loaded Transformer weights from {checkpoint_path}")

    def forward_convnext(self, x):
        for i in range(4):
            x = self.ConvNeXt.downsample_layers[i](x)
            x = self.ConvNeXt.stages[i](x)
        # Reshape from (batch, channels, 7, 7) to (batch, 196, 192)
        return torch.reshape(x, (x.shape[0], 192, 196)).permute(0, 2, 1)

    def forward_transformer(self, x):
        return self.Transformer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid model.
        Args:
            x: Input tensor (batch, 3, 224, 224)
        Returns:
            Age predictions (batch, 1)
        """
        x = self.forward_convnext(x)  # (batch, 196, 192)

        return self.forward_transformer(x)


class AgeHybridModel(nn.Module):
    """
    Wrapper class for the hybrid model that matches the checkpoint structure.
    This class follows the same pattern as in the author's model.py.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        # Extract configs for each component
        convnext_cfg = cfg.get("convnext", {})
        transformer_cfg = cfg.get("transformer", {})

        # Create the hybrid model
        self.model = ConvNeXTTransformer(
            convnext_cfg=convnext_cfg,
            transformer_cfg=transformer_cfg,
        )

        # Load pretrained weights if specified
        if convnext_cfg.get("weights"):
            self.model.load_convnext_weights(cfg.convnext.weights)

        if transformer_cfg.get("weights"):
            self.model.load_transformer_weights(cfg.transformer.weights)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "AgeHybridModel":
        """Create hybrid model from configuration."""
        return cls(cfg)

    def forward(self, x):
        return self.model(x)
