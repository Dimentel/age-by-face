from typing import Literal

import lightning as l
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig

from age_by_face.data.dataset import AgeDataModule
from age_by_face.inference.infer import _resolve_ckpt_path
from age_by_face.models.architecture import build_model
from age_by_face.models.module import AgeRegressionModule


def evaluate(
    cfg: DictConfig,
    split: Literal["train", "val", "test"] = "test",
    verbose: bool = True,
) -> dict[str, float]:
    """
    Evaluate a trained model on a specified dataset split.

    This function loads a model from a checkpoint (resolved via _resolve_ckpt_path),
    sets up the appropriate dataloader for the requested split, and computes
    evaluation metrics using a Lightning Trainer in test mode.

    Args:
        cfg: Hydra configuration object containing model, data, and checkpoint settings.
        split: Which dataset split to evaluate on. Must be one of "train", "val", or "test".
               Defaults to "test".
        verbose: If True, prints progress bar and final results. Defaults to True.

    Returns:
        Dictionary containing evaluation metrics. Typical keys include:
        - 'test_loss': final MSE loss
        - 'test_mae': Mean Absolute Error
        - 'test_mape': Mean Absolute Percentage Error
        - 'test_mse': Mean Squared Error
        (keys may be prefixed with 'test' regardless of split due to Lightning's test step)

    Example:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        >>> metrics = evaluate(cfg, split="val")
        >>> print(f"Validation MAE: {metrics['test_mae']:.2f}")

    Notes:
        - The checkpoint path is resolved using _resolve_ckpt_path from infer module:
          best.ckpt → last.ckpt → cfg.ckpt_path
        - The model architecture must match the one used during training
        - Metrics are computed using the test_step method of AgeRegressionModule
    """

    print(f"DEBUG: cfg.ckpt_path = {getattr(cfg, 'ckpt_path', None)}")
    print(f"DEBUG: cfg.training.checkpoint.dirpath = {cfg.training.checkpoint.dirpath}")

    # Setup datamodule
    datamodule = AgeDataModule(cfg.dataset)

    # Setup appropriate split
    if split == "train":
        datamodule.setup("fit")
        dataloader = datamodule.train_dataloader()
    elif split == "val":
        datamodule.setup("validate")
        dataloader = datamodule.val_dataloader()
    else:
        datamodule.setup("test")
        dataloader = datamodule.test_dataloader()

    # Путь к чекпойнту: best.ckpt -> last.ckpt -> cfg.ckpt_path
    checkpoint_path = _resolve_ckpt_path(cfg)
    print(f"DEBUG: resolved path = {checkpoint_path}")

    if verbose:
        print(f"Loading model from: {checkpoint_path}")

    # Build model architecture
    model = build_model(cfg.model)

    # Load checkpoint
    module = AgeRegressionModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model=model,
        cfg=cfg,
        map_location="cpu",
    )
    logger = (
        CSVLogger(
            "logs",
            name="eval",
            version=f"{cfg.model.type}_{checkpoint_path.stem}_{cfg.dataset.name}_{split}_{cfg.dataset.target_age}",
        )
        if verbose
        else False
    )

    # Create trainer for evaluation
    trainer = l.Trainer(
        accelerator="auto",
        devices="auto",
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=verbose,
    )

    # Run evaluation
    if verbose:
        print(f"Evaluating on {split} set...")

    results = trainer.test(module, dataloaders=dataloader, verbose=verbose)

    # Extract metrics
    metrics = results[0] if results else {}

    if verbose:
        print("\n" + "=" * 50)
        print(f"Evaluation Results on {split.upper()} set:")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("=" * 50)

    return metrics
