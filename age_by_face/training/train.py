import shutil
import subprocess
from pathlib import Path

import lightning as l
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from omegaconf import DictConfig

from age_by_face.data.dataset import AgeDataModule
from age_by_face.models.architecture import build_model
from age_by_face.models.module import AgeRegressionModule
from age_by_face.utils.backbone import freeze_backbone
from age_by_face.utils.download_data import download_data_dvc


def get_git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    try:
        # exit 0 => чисто; non-zero => есть незакоммиченные изменения
        subprocess.check_call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dirty = False
    except subprocess.CalledProcessError:
        dirty = True
    except Exception:
        dirty = False

    return commit, dirty


def _create_module(cfg: DictConfig) -> AgeRegressionModule:
    """Create or load module based on config."""
    checkpoint_path = getattr(cfg.model, "checkpoint_path", None)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Fine-tuning mode: loading from {checkpoint_path}")
        model = build_model(cfg.model)
        module = AgeRegressionModule.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path), model=model, cfg=cfg, map_location="cpu"
        )
        # Update hyperparameters
        module.hparams["optimizer"] = {
            "lr": float(cfg.training.optimizer.lr),
            "weight_decay": float(cfg.training.optimizer.weight_decay),
            "betas": tuple(cfg.training.optimizer.betas),
        }
        module.hparams["scheduler"] = {
            "mode": str(getattr(cfg.training.lr_scheduler, "mode", "min")),
            "factor": float(getattr(cfg.training.lr_scheduler, "factor", 0.5)),
            "patience": int(getattr(cfg.training.lr_scheduler, "patience", 5)),
        }
        # Freeze backbone if needed
        if getattr(cfg.model, "freeze_backbone", False):
            freeze_backbone(module.model)
            print("Backbone frozen for fine-tuning")
    else:
        print("Training from scratch")
        model = build_model(cfg.model)
        module = AgeRegressionModule(model=model, cfg=cfg)

    return module


def _setup_logger(cfg: DictConfig, git_commit: str, git_dirty: bool):
    log_cfg = getattr(cfg, "logging", None)
    if log_cfg and hasattr(log_cfg, "mlflow") and bool(getattr(log_cfg.mlflow, "enabled", False)):
        mlflow_cfg = log_cfg.mlflow
        tags = dict(getattr(mlflow_cfg, "tags", {}))
        # MLflow's tags should be str
        tags.update({"git_commit": str(git_commit), "git_dirty": "true" if git_dirty else "false"})
        logger = MLFlowLogger(
            tracking_uri=str(mlflow_cfg.tracking_uri),
            experiment_name=str(mlflow_cfg.experiment_name),
            run_name=str(mlflow_cfg.run_name),
            tags=tags,
            log_model=bool(getattr(mlflow_cfg, "log_model", False)),
        )
    else:
        # TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir="tb_logs", name=str(getattr(cfg.project, "name", "run"))
        )
        if isinstance(logger, TensorBoardLogger):
            logger.log_hyperparams({"git_commit": str(git_commit), "git_dirty": bool(git_dirty)})

    return logger


def _setup_callbacks(cfg: DictConfig) -> list:
    """Setup training callbacks."""
    # Callbacks
    callbacks = []

    # Early stopping
    es_cfg = getattr(cfg.training, "early_stopping", None)
    if es_cfg and bool(getattr(es_cfg, "enabled", False)):
        callbacks.append(
            EarlyStopping(
                monitor=str(getattr(es_cfg, "monitor", "val_loss")),
                patience=int(getattr(es_cfg, "patience", 10)),
                mode=str(getattr(es_cfg, "mode", "min")),
                verbose=bool(getattr(es_cfg, "verbose", False)),
            )
        )

    # Checkpointing
    ckpt_cfg = getattr(cfg.training, "checkpoint", None)
    if ckpt_cfg:
        ckpt_cb = ModelCheckpoint(
            monitor=str(getattr(ckpt_cfg, "monitor", "val_loss")),
            mode=str(getattr(ckpt_cfg, "mode", "min")),
            save_top_k=int(getattr(ckpt_cfg, "save_top_k", 1)),
            save_last=bool(getattr(ckpt_cfg, "save_last", True)),
            every_n_epochs=int(getattr(ckpt_cfg, "every_n_epochs", 1)),
            dirpath=str(getattr(ckpt_cfg, "dirpath", "checkpoints")),
            filename=str(
                getattr(ckpt_cfg, "filename", "{cfg.dataset.target_age}_{epoch}-{val_loss:.2f}")
            ),
        )
        callbacks.append(ckpt_cb)

    # LR monitor (control ReduceLROnPlateau)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    return callbacks


def train(cfg: DictConfig) -> None:
    # Reproducibility
    l.seed_everything(int(getattr(cfg, "seed", 5)), workers=True)

    # DVC: гарантируем наличие данных локально
    download_data_dvc(cfg)

    # Data
    datamodule = AgeDataModule(cfg.dataset)

    # Model
    module = _create_module(cfg)

    # Git info
    git_commit, git_dirty = get_git_info()

    # Logger
    logger = _setup_logger(cfg, git_commit, git_dirty)

    # Callbacks
    callbacks = _setup_callbacks(cfg)

    ckpt_cb = None
    if len(callbacks) > 1 and isinstance(callbacks[1], ModelCheckpoint):
        ckpt_cb = callbacks[1]

    # Trainer
    trainer = l.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        accumulate_grad_batches=int(getattr(cfg.training, "accumulate_grad_batches", 1)),
        precision=str(getattr(cfg.training, "precision", "32-true")),
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
    )

    # Fit
    trainer.fit(module, datamodule=datamodule)

    # Логируем лучший чекпоинт как артефакт MLflow и копируем в стабильное имя в каталоге чекпоинтов
    if not isinstance(logger, TensorBoardLogger) and ckpt_cb and ckpt_cb.best_model_path:
        try:
            best_src = Path(ckpt_cb.best_model_path)
            best_dir = Path(str(getattr(cfg.training.checkpoint, "dirpath", "checkpoints")))
            best_dir.mkdir(parents=True, exist_ok=True)
            best_dst = best_dir / "best.ckpt"  # fixed name for inference
            shutil.copy2(best_src, best_dst)

            # Log in directory "checkpoints" of current MLflow run
            logger.experiment.log_artifact(
                logger.run_id, str(best_dst), artifact_path="checkpoints"
            )
        except Exception as e:
            print(f"Warning: failed to log best checkpoint to MLflow: {e}")
