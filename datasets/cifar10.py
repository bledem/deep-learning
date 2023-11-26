"""
More at https://lightning.ai/docs/pytorch/stable/data/datamodule.html
"""
import logging
from pathlib import Path

import lightning.pytorch as pl
import torch
import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Create a logger
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.INFO)

_DEFAULT_BATCH_SIZE = 32
# Upscaling the image to match with ImageNet size.
_DEFAULT_RESIZE_SIZE = 227


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = _DEFAULT_BATCH_SIZE):
        super().__init__()
        self.generator = torch.Generator().manual_seed(42)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((_DEFAULT_RESIZE_SIZE, _DEFAULT_RESIZE_SIZE)),
                cifar10_normalization(),
            ]
        )
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.Resize((_DEFAULT_RESIZE_SIZE, _DEFAULT_RESIZE_SIZE)),
                cifar10_normalization(),
            ]
        )

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, train=True, transform=self.train_transform, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.val_transform, download=True)

    def setup(self, stage: str):
        """Is called from every process across all nodes.
        It also uses every GPUs to perform data processing and state assignement.
        `teardown` is its counterpart used to clean the states.
        """
        logger.info(f"Stage: {stage}")
        if stage == "test" or stage == "predict":
            self.cifar_test = torchvision.datasets.CIFAR10(
                self.data_dir, train=False, download=True, transform=self.train_transform
            )
        elif stage == "fit" or stage == "validate":
            self.cifar_train_val = torchvision.datasets.CIFAR10(
                self.data_dir, train=False, transform=self.val_transform, download=True
            )
            self.cifar_train, self.cifar_val = random_split(self.cifar_train_val, [0.7, 0.3], generator=self.generator)

    def train_dataloader(self) -> DataLoader:
        """Called by Trainer `.fit` method"""
        return DataLoader(self.cifar_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Called by Trainer `validate()` and `validate()` method."""
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Called by Trainer `test()` method."""
        return DataLoader(self.cifar_test, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader:
        """Called by Trainer `predict()` method. Use the same data as the test_dataloader."""
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=3)
