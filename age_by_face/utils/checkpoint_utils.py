import torch
from omegaconf import DictConfig

from age_by_face.models.architecture import build_model
from age_by_face.models.module import AgeRegressionModule


def load_model(cfg: DictConfig, checkpoint_path: str) -> AgeRegressionModule:
    """
    Load model from checkpoint with special handling for hybrid architecture.
    Args:
        cfg: Hydra configuration
        checkpoint_path: Path to checkpoint file
    Returns:
        Loaded AgeRegressionModule
    """
    model = build_model(cfg.model)

    if cfg.model.type == "hybrid":
        # Special handling for hybrid model (needs key transformation)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        # Transform keys: add one 'model.' prefix
        adapted_state_dict = {}
        for k, v in state_dict.items():
            # model. + model.ConvNeXt... = model.model.ConvNeXt...
            if k.startswith("model.model.model."):
                new_k = "model.model." + k[18:]  # убираем один 'model.'
            elif k.startswith("model.model."):
                # model.model. - ok
                new_k = k
            elif k.startswith("model."):
                # model. -> model.model.
                new_k = "model." + k
            else:
                new_k = k
            adapted_state_dict[new_k] = v
        module = AgeRegressionModule(model=model, cfg=cfg)
        module.load_state_dict(adapted_state_dict)

        print("Loaded hybrid checkpoint with key transformation")
    else:
        # Normal loading for all other models
        module = AgeRegressionModule.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            model=model,
            cfg=cfg,
            map_location="cpu",
        )

    return module
