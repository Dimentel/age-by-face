import lightning as l
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig

from age_by_face.data.dataset import AgeDataModule
from age_by_face.inference.infer import _resolve_ckpt_path
from age_by_face.utils.checkpoint_utils import load_model


def evaluate(cfg: DictConfig) -> dict[str, float]:
    """
    Evaluate a trained model on a specified dataset split.

    This function loads a model from a checkpoint (resolved via _resolve_ckpt_path),
    sets up the appropriate dataloader for the requested split, and computes
    evaluation metrics using a Lightning Trainer in test mode.

    Args:
        cfg: Hydra configuration object containing model, data, and checkpoint settings.

    Returns:
        Dictionary containing evaluation metrics. Typical keys include:
        - 'test_loss': final MSE loss
        - 'test_mae': Mean Absolute Error
        - 'test_mape': Mean Absolute Percentage Error
        - 'test_mse': Mean Squared Error
        (keys may be prefixed with 'test' regardless of split due to Lightning's test step)

    Notes:
        - The checkpoint path is resolved using _resolve_ckpt_path from infer module:
          best.ckpt → last.ckpt → cfg.ckpt_path
        - The model architecture must match the one used during training
        - Metrics are computed using the test_step method of AgeRegressionModule
    """

    # Setup datamodule
    datamodule = AgeDataModule(cfg.dataset)

    split = cfg.eval.split
    verbose = cfg.eval.verbose
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

    if verbose:
        print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    module = load_model(cfg, checkpoint_path)
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
