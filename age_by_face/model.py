import torch
from torch import nn
from torchvision import models


class AgeResNet18(nn.Module):
    """Baseline model for age regression using ResNet18 backbone."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Load ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Remove the original fully connected layer
        # Keep layers up to avgpool (but we'll replace avgpool)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Original ResNet18 has 512 output channels before avgpool
        in_features = 512

        # Custom layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Hidden layer
        self.fc1 = nn.Linear(in_features, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Output layer
        self.fc2 = nn.Linear(512, 1)
        self.relu2 = nn.ReLU()  # Для возрастов ≥ 0

        # Initialize weights for custom layers
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for custom fully connected layers."""
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone (ResNet18 without original FC)
        features = self.backbone(x)

        # Global pooling
        pooled = self.avg_pool(features)
        flattened = self.flatten(pooled)

        # Hidden layer
        hidden = self.fc1(flattened)
        hidden = self.relu1(hidden)
        hidden = self.dropout1(hidden)

        # Output layer with ReLU for non-negative ages
        output = self.fc2(hidden)

        return self.relu2(output)


class AgeResNet50(nn.Module):
    """Main model for age regression using ResNet50 backbone."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        # Load ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Remove the original fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # ResNet50 has 2048 output channels before avgpool
        in_features = 2048

        # Custom layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Hidden layer
        self.fc1 = nn.Linear(in_features, 2048)

        # Optional batch normalization
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(2048)
        else:
            self.bn1 = nn.Identity()

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Output layer
        self.fc2 = nn.Linear(2048, 1)
        self.relu2 = nn.ReLU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for custom fully connected layers."""
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)

        if hasattr(self.bn1, "weight"):
            nn.init.ones_(self.bn1.weight)
            nn.init.zeros_(self.bn1.bias)

        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone (ResNet50 without original FC)
        features = self.backbone(x)

        # Global pooling
        pooled = self.avg_pool(features)
        flattened = self.flatten(pooled)

        # Hidden layer with optional batch norm
        hidden = self.fc1(flattened)
        hidden = self.bn1(hidden)
        hidden = self.relu1(hidden)
        hidden = self.dropout1(hidden)

        # Output layer with ReLU for non-negative ages
        output = self.fc2(hidden)

        return self.relu2(output)
