import lightning as l
import torch
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError


class AgeRegressionModule(l.LightningModule):
    """Lightning Module for age regression from face images."""

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        Args:
            model: PyTorch model for age regression
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for AdamW optimizer
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

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
            on_step=True,
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
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
