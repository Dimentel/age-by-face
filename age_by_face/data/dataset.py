import lightning as l
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from age_by_face.utils.path_utils import ensure_path
from age_by_face.utils.types import Directory, File


class AgeDataset(Dataset):
    """Custom dataset for age regression from face images with CSV annotations."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Directory,
        transform: transforms.Compose | None = None,
    ):
        """
        Args:
            df: DataFrame containing columns with file names and ages (real or apparent)
            images_dir: Directory containing face images
            transform: Optional torchvision transforms to apply
        """
        self.images_dir = images_dir
        self.transform = transform

        # Create full image paths
        self.image_paths = [self.images_dir / filename for filename in df.iloc[:, 0]]

        # Convert ages to tensors (float for regression)
        self.ages = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).unsqueeze(
            1
        )  # Shape: [n_samples, 1]

    def __len__(self) -> int:
        return len(self.image_paths)

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


def init_dataset(  # noqa: PLR0913
    csv_path: File,
    images_dir: Directory,
    columns: list,
    transformer: transforms.Compose,
    filename_suffix: str = "",
    target_age_shift: int = 0,
) -> AgeDataset:
    """Initialize age dataset with appropriate transforms.

    Args:
        csv_path: Path to CSV file
        images_dir: Directory with images
        columns: List of column names
        transformer: Optional torchvision transforms to apply

    Returns:
        AgeDataset instance
    """

    # Преобразуем к Path
    csv_path = ensure_path(csv_path)
    images_dir = ensure_path(images_dir)

    # Проверка существования путей
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not images_dir.is_dir():
        raise ValueError(f"Images directory is not directory: {images_dir}")

    # Load annotations
    df = pd.read_csv(csv_path)

    # Validate required columns
    if not set(columns).issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {columns}")
    if filename_suffix:
        df[columns[0]] = df[columns[0]] + filename_suffix
    if target_age_shift:
        df[columns[1]] = df[columns[1]] + target_age_shift

    return AgeDataset(df[columns].copy(), images_dir, transformer)


def init_predict_dataset():
    raise NotImplementedError("Predict stage is not implemented yet")


def init_dataloader(  # noqa: PLR0913
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = True,
) -> DataLoader:
    """Initialize torch dataloader from dataset.

    Args:
        dataset: Dataset for dataloader
        batch_size: Batch size
        shuffle: Whether to shuffle dataset
        num_workers: Number of workers for dataset loading
        pin_memory: Whether to pin memory for faster data transfer to GPU
        persistent_workers: Whether to use persistent workers for faster data loading
        drop_last: Whether to drop last batch if it's smaller than batch_size

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


class AgeDataModule(l.LightningDataModule):
    """LightningDataModule for age regression dataset."""

    def __init__(self, dataset_cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = dataset_cfg

        self.data_dir = ensure_path(self.cfg.data_dir)

        # CSV file names
        self.train_csv = self.data_dir / self.cfg.train_csv
        self.val_csv = self.data_dir / self.cfg.val_csv
        self.test_csv = self.data_dir / self.cfg.test_csv

        # columns
        if self.cfg.target_age == "real":
            self.columns = [self.cfg.file_name_col, self.cfg.real_age_col]
        elif self.cfg.target_age == "apparent":
            self.columns = [self.cfg.file_name_col, self.cfg.apparent_age_col]
        else:
            raise ValueError(f"Unknown age_to_predict: {self.cfg.target_age}")

        # Batch sizes
        self.train_batch_size = self.cfg.train_batch_size
        self.val_batch_size = getattr(self.cfg, "val_batch_size", self.cfg.predict_batch_size)
        self.test_batch_size = getattr(self.cfg, "test_batch_size", self.cfg.predict_batch_size)

        # Other parameters
        self.num_workers = self.cfg.num_workers
        self.persistent_workers = bool(
            getattr(self.cfg, "persistent_workers", self.num_workers > 0)
        )
        self.pin_memory = bool(getattr(self.cfg, "pin_memory", True))

        # Image directories
        self.train_dir = self.data_dir / self.cfg.train_dir_name
        self.val_dir = self.data_dir / self.cfg.val_dir_name
        self.test_dir = self.data_dir / self.cfg.test_dir_name
        self.predict_dir = self.data_dir / self.cfg.predict_dir_name

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.train_transformer = transforms.Compose(
            [
                transforms.Resize(tuple(self.cfg.image_size)),
                transforms.RandomRotation(degrees=self.cfg.random_rotation_angle),
                transforms.RandomHorizontalFlip(p=self.cfg.random_horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=dataset_cfg.normalize_means, std=dataset_cfg.normalize_stds
                ),  # [-1, 1] range
            ]
        )

        self.val_transformer = transforms.Compose(
            [
                transforms.Resize(tuple(self.cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.cfg.normalize_means, std=self.cfg.normalize_stds
                ),  # [-1, 1] range
            ]
        )

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for different stages."""

        if stage == "fit" or stage is None:
            self.train_dataset = init_dataset(
                csv_path=self.train_csv,
                images_dir=self.train_dir,
                columns=self.columns,
                transformer=self.train_transformer,
                filename_suffix=self.cfg.filename_suffix,
                target_age_shift=self.cfg.target_age_shift,
            )
            self.val_dataset = init_dataset(
                csv_path=self.val_csv,
                images_dir=self.val_dir,
                columns=self.columns,
                transformer=self.val_transformer,
                filename_suffix=self.cfg.filename_suffix,
                target_age_shift=self.cfg.target_age_shift,
            )
        elif stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = init_dataset(
                    csv_path=self.val_csv,
                    images_dir=self.val_dir,
                    columns=self.columns,
                    transformer=self.val_transformer,
                    filename_suffix=self.cfg.filename_suffix,
                    target_age_shift=self.cfg.target_age_shift,
                )
        elif stage == "test":
            self.test_dataset = init_dataset(
                csv_path=self.test_csv,
                images_dir=self.test_dir,
                columns=self.columns,
                transformer=self.val_transformer,
                filename_suffix=self.cfg.filename_suffix,
                target_age_shift=self.cfg.target_age_shift,
            )
        elif stage == "predict":
            raise NotImplementedError("Predict stage is not implemented yet")

    def train_dataloader(self) -> DataLoader:
        return init_dataloader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return init_dataloader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return init_dataloader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        raise NotImplementedError("Predict stage is not implemented yet")
