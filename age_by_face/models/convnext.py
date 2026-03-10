"""
ConvNeXT model implementation adapted for age regression.
Based on: https://github.com/facebookresearch/ConvNeXt
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import torch
import torch.nn.functional as f
from omegaconf import DictConfig
from timm.layers import DropPath, trunc_normal_
from torch import nn


class Block(nn.Module):
    """ConvNeXt Block with depthwise conv and layer scale.
    There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C);
        LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as it was found slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input_img = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        return input_img + self.drop_path(x)


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return f.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)

        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXt(nn.Module):
    """ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans  (int): Number of input channels (default: 3)
        depths (tuple(int)): Number of blocks at each stage (default: (3, 3, 9, 3))
        dims (tuple(int)): Feature dimensions at each stage (default: (96, 192, 384, 768))
        drop_path_rate (float): Stochastic depth rate (default: 0.)
        layer_scale_init_value (float): Init value for Layer Scale (default: 1e-6)
        head_init_scale (float): Init scaling value for classifier weights (default: 1.)
    """

    def __init__(  # noqa: PLR0913
        self,
        in_chans: int = 3,
        num_classes=1000,
        depths: tuple[int] = (3, 3, 9, 3),
        dims: tuple[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers

        # Stem
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # Downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)  # dummy head, will be replaced

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)

        return self.head(x)


class AgeConvNeXt(nn.Module):
    """
    ConvNeXT model adapted for age regression.

    Supports:
        - Loading pretrained weights from timm or checkpoint
        - Configurable hidden layer (fc parameter)
        - Single output for regression
    """

    def __init__(  # noqa: PLR0913
        self,
        in_chans: int = 3,
        num_classes: int = 1,
        depths: tuple[int] = (3, 3, 9, 3),
        dims: tuple[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        fc_hidden_size: int = 256,  # size of hidden layer, if None uses single linear layer
        output_activation: str | None = "relu",
    ):
        super().__init__()

        # Create backbone
        convnext = ConvNeXt(
            in_chans=in_chans,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            head_init_scale=head_init_scale,
        )

        self.downsample_layers = convnext.downsample_layers
        self.stages = convnext.stages
        self.norm = convnext.norm

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 224, 224)
            features = self.forward_features(dummy)
            self.output_size = features.shape[1:]

        # Regression head
        if fc_hidden_size:
            if output_activation:
                self.fc = nn.Sequential(
                    nn.Linear(self.output_size[-1], fc_hidden_size),
                    nn.Linear(fc_hidden_size, num_classes),
                    nn.ReLU(inplace=True),
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(self.output_size[-1], fc_hidden_size),
                    nn.Linear(fc_hidden_size, num_classes),
                )
        elif output_activation:
            self.fc = nn.Sequential(
                nn.Linear(self.output_size[-1], num_classes),
                nn.ReLU(inplace=True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.output_size[-1], num_classes),
            )

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "AgeConvNeXt":
        """Create model from configuration."""
        return cls(
            in_chans=cfg.get("in_chans", 3),
            num_classes=cfg.get("output_dim", 1),
            depths=cfg.get("depths", None),
            dims=cfg.get("dims", None),
            drop_path_rate=cfg.get("drop_path_rate", 0.0),
            layer_scale_init_value=cfg.get("layer_scale_init_value", 1e-6),
            head_init_scale=cfg.get("head_init_scale", 1.0),
            fc_hidden_size=cfg.get("fc", 256),
            output_activation=cfg.get("output_activation", None),  # null, "relu"
        )

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        return self.fc(x)
