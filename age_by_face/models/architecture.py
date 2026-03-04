from omegaconf import DictConfig
from torch import nn

from age_by_face.models.resnet import AgeResNet18, AgeResNet50

# Future models will be imported here


def build_model(model_cfg: DictConfig) -> nn.Module:
    """
    Build model from configuration.
    Creates only the architecture, does not load weights.
    """

    model_type = str(getattr(model_cfg, "type", "resnet18")).lower()
    if model_type == "resnet18":
        model = AgeResNet18.from_cfg(model_cfg)
    elif model_type == "resnet50":
        model = AgeResNet50.from_cfg(model_cfg)
    # Future models will be added here with elif
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model
