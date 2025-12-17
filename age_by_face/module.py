import lightning as l
import torch
from omegaconf import DictConfig
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError


class AgeRegressionModule(l.LightningModule):
    """Lightning Module for age regression from face images."""

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DictConfig,
    ):
        """
        Args:
            model: PyTorch model for age regression
            cfg: Hydra configuration DictConfig
        """
        super().__init__()
        self.model = model
        self.cfg = cfg

        # Loss function for regression
        self.criterion = torch.nn.MSELoss()

        # Metrics for validation
        self.val_mae = MeanAbsoluteError()
        self.val_mape = MeanAbsolutePercentageError()
        self.val_mse = MeanSquaredError()

        # Metrics for testing
        self.test_mae = MeanAbsoluteError()
        self.test_mape = MeanAbsolutePercentageError()
        self.test_mse = MeanSquaredError()

        # Save hyperparameters to checkpoint
        self.save_hyperparameters(
            {
                "optimizer": {
                    "lr": float(self.cfg.training.optimizer.lr),
                    "weight_decay": float(self.cfg.training.optimizer.weight_decay),
                    "betas": tuple(self.cfg.training.optimizer.betas),
                },
                "scheduler": {
                    "mode": str(getattr(self.cfg.training.lr_scheduler, "mode", "min")),
                    "factor": float(getattr(self.cfg.training.lr_scheduler, "factor", 0.5)),
                    "patience": int(getattr(self.cfg.training.lr_scheduler, "patience", 5)),
                },
                "model_type": str(getattr(self.cfg.model, "type", "unknown")),
                "image_size": tuple(getattr(self.cfg.dataset, "image_size", (224, 224))),
                "seed": int(getattr(self.cfg, "seed", 0)),
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            inputs: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Age predictions tensor of shape [batch_size, 1]
        """
        return self.model(inputs)

    def training_step(self, batch):
        """Training step with MSE loss."""
        images, ages = batch

        # Forward pass
        predictions = self(images)

        # Calculate loss
        loss = self.criterion(predictions, ages)

        # Log training loss
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch):
        """Validation step with MSE, MAE, and MAPE metrics."""
        images, ages = batch

        # Forward pass
        predictions = self(images)

        # Calculate loss
        loss = self.criterion(predictions, ages)

        # Update metrics
        self.val_mse(predictions, ages)
        self.val_mae(predictions, ages)
        self.val_mape(predictions, ages)

        # Log validation loss
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

        return loss

    def on_validation_epoch_end(self):
        """Log validation metrics at the end of epoch."""
        # Compute metrics
        val_mse = self.val_mse.compute()
        val_mae = self.val_mae.compute()
        val_mape = self.val_mape.compute()

        # Log metrics
        self.log_dict(
            {
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_mape": val_mape,
            },
            prog_bar=True,
            logger=True,
        )

        # Reset metrics for next epoch
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_mape.reset()

    def test_step(self, batch):
        """Test step with MSE, MAE, and MAPE metrics."""
        images, ages = batch

        # Forward pass
        predictions = self(images)

        # Calculate loss
        loss = self.criterion(predictions, ages)

        # Update metrics
        self.test_mse(predictions, ages)
        self.test_mae(predictions, ages)
        self.test_mape(predictions, ages)

        # Log test loss
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def on_test_epoch_end(self):
        """Log test metrics at the end of epoch."""
        # Compute metrics
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        test_mape = self.test_mape.compute()

        # Log metrics
        self.log_dict(
            {
                "test_mse": test_mse,
                "test_mae": test_mae,
                "test_mape": test_mape,
            },
            prog_bar=True,
            logger=True,
        )

        # Reset metrics
        self.test_mse.reset()
        self.test_mae.reset()
        self.test_mape.reset()

    def predict_step(self, batch):
        """Prediction step for inference."""
        images = batch[0] if isinstance(batch, (list, tuple)) else batch

        return self(images)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Получаем параметры оптимизатора из конфига
        optimizer_cfg = self.cfg.training.optimizer
        # Создаём оптимизатор
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(optimizer_cfg.lr),
            weight_decay=float(optimizer_cfg.weight_decay),
            betas=tuple(optimizer_cfg.betas),
        )

        # Проверяем есть ли scheduler в конфиге
        scheduler_cfg = getattr(self.cfg.training, "lr_scheduler", None)
        if scheduler_cfg is None:
            return optimizer

        # Создаём scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.mode),
            factor=float(scheduler_cfg.factor),
            patience=int(scheduler_cfg.patience),
            threshold=float(getattr(scheduler_cfg, "threshold", 1e-4)),
            threshold_mode=str(getattr(scheduler_cfg, "threshold_mode", "rel")),
            cooldown=int(getattr(scheduler_cfg, "cooldown", 0)),
            min_lr=float(getattr(scheduler_cfg, "min_lr", 0.0)),
            verbose=bool(getattr(scheduler_cfg, "verbose", False)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": getattr(scheduler_cfg, "monitor", "val_loss"),
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_age_statistics(self) -> dict:
        """Get statistics about predictions (for analysis).

        Returns:
            Dictionary with mean predicted age, std, etc.
        """
        # This method can be used for analysis after training
        pass
