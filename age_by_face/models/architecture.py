import torch
from omegaconf import DictConfig
from torch import nn
from torchvision import models


def _make_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    if name == "gelu":
        return nn.GELU()
    if name == "softplus":
        return nn.Softplus()
    if name in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def _init_head_weights(module: nn.Module, method: str, act: str) -> None:
    method = (method or "kaiming").lower()
    act_l = (act or "relu").lower()
    # For Kaiming set nonlinearity
    if act_l == "leaky_relu":
        nonlin = "leaky_relu"
        a = 0.01
    elif act_l == "relu":
        nonlin = "relu"
        a = 0.0
    else:
        nonlin = "linear"
        a = 0.0

    for m in module.modules():
        if isinstance(m, nn.Linear):
            if method == "kaiming":
                nn.init.kaiming_normal_(m.weight, a=a, mode="fan_in", nonlinearity=nonlin)
            elif method == "xavier":
                nn.init.xavier_normal_(m.weight)
            else:
                raise ValueError(f"Unsupported init method: {method}")
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _build_head(  # noqa: PLR0913
    in_features: int,
    hidden_layers: list[int],
    activation: str,
    output_dim: int,
    output_activation: str | None,
    dropout_rate: float,
    use_batch_norm: bool,
    init_method: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_features

    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(_make_activation(activation))
        if dropout_rate and dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        prev = h

    layers.append(nn.Linear(prev, output_dim))
    if output_activation and output_activation.lower() not in {"none", "identity"}:
        layers.append(_make_activation(output_activation))

    head = nn.Sequential(*layers)
    _init_head_weights(head, init_method, activation)
    return head


class AgeResNet18(nn.Module):
    """Baseline model for age regression using ResNet18 backbone."""

    def __init__(  # noqa: PLR0913
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
        output_dim: int = 1,
        output_activation: str | None = "relu",
        init_method: str = "kaiming",
        use_batch_norm: bool = True,
    ):
        super().__init__()

        # Load ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Remove the original fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Original ResNet18 has 512 output channels before avgpool
        in_features = 512

        # Custom layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        sizes = hidden_layers or [512]
        self.head = _build_head(
            in_features=in_features,
            hidden_layers=sizes,
            activation=activation,
            output_dim=output_dim,
            output_activation=output_activation,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            init_method=init_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone (ResNet18 without original FC)
        features = self.backbone(x)

        # Global pooling
        pooled = self.avg_pool(features)
        flattened = self.flatten(pooled)

        return self.head(flattened)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "AgeResNet18":
        return cls(
            pretrained=bool(getattr(cfg, "pretrained", True)),
            dropout_rate=float(getattr(cfg, "dropout_rate", 0.3)),
            freeze_backbone=bool(getattr(cfg, "freeze_backbone", False)),
            hidden_layers=list(getattr(cfg, "hidden_layers", [512])),
            activation=str(getattr(cfg, "activation", "relu")),
            output_dim=int(getattr(cfg, "output_dim", 1)),
            output_activation=str(getattr(cfg, "output_activation", "relu")),
            init_method=str(getattr(cfg, "init_method", "kaiming")),
            use_batch_norm=bool(getattr(cfg, "use_batch_norm", True)),
        )


class AgeResNet50(nn.Module):
    """Main model for age regression using ResNet50 backbone."""

    def __init__(  # noqa: PLR0913
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        use_batch_norm: bool = True,
        freeze_backbone: bool = False,
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
        output_dim: int = 1,
        output_activation: str | None = "relu",
        init_method: str = "kaiming",
    ):
        super().__init__()

        # Load ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Remove the original fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # ResNet50 has 2048 output channels before avgpool
        in_features = 2048

        # Custom layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        sizes = hidden_layers or [2048]
        self.head = _build_head(
            in_features=in_features,
            hidden_layers=sizes,
            activation=activation,
            output_dim=output_dim,
            output_activation=output_activation,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            init_method=init_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone (ResNet50 without original FC)
        features = self.backbone(x)

        # Global pooling
        pooled = self.avg_pool(features)
        flattened = self.flatten(pooled)

        return self.head(flattened)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "AgeResNet50":
        return cls(
            pretrained=bool(getattr(cfg, "pretrained", True)),
            dropout_rate=float(getattr(cfg, "dropout_rate", 0.4)),
            use_batch_norm=bool(getattr(cfg, "use_batch_norm", True)),
            freeze_backbone=bool(getattr(cfg, "freeze_backbone", False)),
            hidden_layers=list(getattr(cfg, "hidden_layers", [2048])),
            activation=str(getattr(cfg, "activation", "relu")),
            output_dim=int(getattr(cfg, "output_dim", 1)),
            output_activation=str(getattr(cfg, "output_activation", "relu")),
            init_method=str(getattr(cfg, "init_method", "kaiming")),
        )


def build_model(model_cfg: DictConfig) -> nn.Module:
    """Фабрика: создаёт модель по cfg.model.type."""
    model_type = str(getattr(model_cfg, "type", "resnet18")).lower()
    if model_type == "resnet18":
        return AgeResNet18.from_cfg(model_cfg)
    if model_type == "resnet50":
        return AgeResNet50.from_cfg(model_cfg)
    raise ValueError(f"Unsupported model type: {model_type}")
