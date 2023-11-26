"""
More at https://lightning.ai/docs/pytorch/stable/data/datamodule.html
"""
import logging
from pathlib import Path

import lightning.pytorch as pl
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Create a logger
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.INFO)

_DEFAULT_BATCH_SIZE = 32
# Upscaling the image to match with ImageNet size.
_DEFAULT_RESIZE_SIZE = 227


class Caltech101(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        image_size: int = _DEFAULT_RESIZE_SIZE,
    ):
        super().__init__()
        self.generator = torch.Generator().manual_seed(42)
        self.data_dir = data_dir
        self.batch_size = batch_size
        # Training transform
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Add any additional transforms you wish to apply only during training
            ]
        )

        # Validation/Testing transform
        self.val_test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def prepare_data(self):
        torchvision.datasets.Caltech101(self.data_dir, download=True)

    def setup(self, stage: str):
        """Is called from every process across all nodes.
        It also uses every GPUs to perform data processing and state assignement.
        `teardown` is its counterpart used to clean the states.
        """
        logger.info(f"Stage: {stage}")
        dataset = torchvision.datasets.Caltech101(self.data_dir, download=True)
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, [0.7, 0.15, 0.15], generator=self.generator
        )

    def train_dataloader(self) -> DataLoader:
        """Called by Trainer `.fit` method"""
        self.data_train.dataset.transform = self.train_transform
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Called by Trainer `validate()` and `validate()` method."""
        self.data_val.dataset.transform = self.val_test_transform
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Called by Trainer `test()` method."""
        self.data_test.dataset.transform = self.val_test_transform
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader:
        """Called by Trainer `predict()` method. Use the same data as the test_dataloader."""
        self.data_test.dataset.transform = self.val_test_transform
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=3)
