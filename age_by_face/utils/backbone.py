from torch import nn


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all parameters except the final regression head.
    Works with any model architecture.
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze final layers (by common naming patterns)
    unfrozen_count = 0
    for name, param in model.named_parameters():
        # Common names for final layers in different architectures
        if any(
            name.endswith(suffix)
            for suffix in [
                "fc2.weight",
                "fc2.bias",  # Our ResNet heads
                "fc.weight",
                "fc.bias",  # Standard ResNet
                "head.weight",
                "head.bias",  # Common pattern
                "classifier.weight",
                "classifier.bias",  # Another pattern
                "mlp.head.weight",
                "mlp.head.bias",  # ViT style
            ]
        ):
            param.requires_grad = True
            unfrozen_count += 1

    print(f"Backbone frozen for fine-tuning. Unfrozen layers: {unfrozen_count}")
