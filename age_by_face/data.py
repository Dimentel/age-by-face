from pathlib import Path

import lightning as l
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from age_by_face.types import Directory, File
from age_by_face.utils.path_utils import ensure_path


class AgeDataset(Dataset):
    """Custom dataset for age regression from face images with CSV annotations."""

    def __init__(
        self,
        csv_path: str | File,
        images_dir: str | Directory,
        transform: transforms.Compose | None = None,
    ):
        """
        Args:
            csv_path: Path to CSV file with columns 'file_name' and 'real_age'
            images_dir: Directory containing face images
            transform: Optional torchvision transforms to apply
        """
        self.csv_path = ensure_path(csv_path)
        self.images_dir = ensure_path(images_dir)

        # Проверка существования путей
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.transform = transform

        # Load annotations
        self.df = pd.read_csv(csv_path)

        # Validate required columns
        required_columns = {"file_name", "real_age"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        # Create full image paths
        self.image_paths = [self.images_dir / filename for filename in self.df["file_name"]]

        # Convert ages to tensors (float for regression)
        self.ages = torch.tensor(self.df["real_age"].values, dtype=torch.float32).unsqueeze(
            1
        )  # Shape: [n_samples, 1]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image, age) pair."""
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        # Get corresponding age
        age = self.ages[idx]

        return image, age


def get_train_transforms() -> transforms.Compose:
    """Get transforms for training data with augmentations."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1] range
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """Get transforms for validation/test/predict data (no augmentations)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1] range
        ]
    )


def init_dataset(
    csv_path: str | File, images_dir: str | Directory, is_train: bool = False
) -> AgeDataset:
    """Initialize age dataset with appropriate transforms.

    Args:
        csv_path: Path to CSV file
        images_dir: Directory with images
        is_train: Whether to use training transforms (with augmentations)

    Returns:
        AgeDataset instance
    """
    # Преобразуем к Path
    csv_path = ensure_path(csv_path)
    images_dir = ensure_path(images_dir)

    transformer = get_train_transforms() if is_train else get_val_transforms()

    return AgeDataset(csv_path, images_dir, transformer)


def init_dataloader(
    dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4
) -> DataLoader:
    """Initialize torch dataloader from dataset.

    Args:
        dataset: Dataset for dataloader
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )


def init_predict_dataset(
    csv_path: str | File, images_dir: str | Directory, is_train: bool = False
) -> AgeDataset:
    raise NotImplementedError("Predict stage is not implemented yet")


class AgeDataModule(l.LightningDataModule):
    """LightningDataModule for age regression dataset."""

    def __init__(  # noqa: PLR0913
        self,
        data_dir: str | Directory,
        train_csv: str = "gt_train.csv",
        val_csv: str = "gt_valid.csv",
        test_csv: str = "gt_test.csv",
        train_batch_size: int = 32,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        num_workers: int = 4,
        train_dir: str = "train",
        val_dir: str = "valid",
        test_dir: str = "test",
        predict_dir: str = "predict",
    ):
        super().__init__()

        self.data_dir = Path(data_dir)

        # CSV file names
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv

        # Batch sizes
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        # Other parameters
        self.num_workers = num_workers

        # Image directories
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.predict_dir = predict_dir

        # Datasets (will be initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            self.train_dataset = init_dataset(
                csv_path=self.data_dir / self.train_csv,
                images_dir=self.data_dir / self.train_dir,
                is_train=True,
            )
            self.val_dataset = init_dataset(
                csv_path=self.data_dir / self.val_csv,
                images_dir=self.data_dir / self.val_dir,
                is_train=False,
            )
        elif stage == "validate" or stage is None:
            if self.val_dataset is None:
                self.val_dataset = init_dataset(
                    csv_path=self.data_dir / self.val_csv,
                    images_dir=self.data_dir / self.val_dir,
                    is_train=False,
                )
        elif stage == "test":
            self.test_dataset = init_dataset(
                csv_path=self.data_dir / self.test_csv,
                images_dir=self.data_dir / self.test_dir,
                is_train=False,
            )
        elif stage == "predict":
            raise NotImplementedError("Predict stage is not implemented yet")

    def train_dataloader(self) -> DataLoader:
        return init_dataloader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return init_dataloader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return init_dataloader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        raise NotImplementedError("Predict stage is not implemented yet")


def test_dataset(data_dir: str | Directory) -> None:
    """Test function to verify dataset loading."""

    # Преобразуем к Path
    data_dir = ensure_path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Example usage
    data_dir = Path(data_dir)
    dataset = init_dataset(
        csv_path=data_dir / "gt_train.csv", images_dir=data_dir / "train", is_train=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(
        f"Age stats: min={dataset.ages.min()}, max={dataset.ages.max()}, "
        f"mean={dataset.ages.mean():.2f}"
    )

    # Show first sample
    img, age = dataset[0]
    print(f"Image shape: {img.shape}, Age: {age.item()}")

    # Denormalize for display
    img_display = img * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    img_display = img_display.permute(1, 2, 0).numpy()

    plt.imshow(img_display)
    plt.title(f"Age: {age.item()}")
    plt.show()
