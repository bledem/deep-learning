"""
More at https://lightning.ai/docs/pytorch/stable/data/datamodule.html
"""
import logging
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# Create a logger
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.INFO)

_DEFAULT_MNIST_BATCH_SIZE = 32
_DEFAULT_RESIZE_SIZE = 32


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = _DEFAULT_MNIST_BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((_DEFAULT_RESIZE_SIZE, _DEFAULT_RESIZE_SIZE)),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        """Ensure we download using one process only on CPU and avoid data corruption when downloading the data.
        It's recommended to avoid creating class attributes `self.*` because the state won't be available for
        other processes.
        """
        MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage: str):
        """Is called from every process across all nodes.
        It also uses every GPUs to perform data processing and state assignement.
        `teardown` is its counterpart used to clean the states.
        """
        logger.info(f"Stage: {stage}")
        if stage == "test" or stage == "predict":
            self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)
        elif stage == "fit" or stage == "validate":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self) -> DataLoader:
        """Called by Trainer `.fit` method"""
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Called by Trainer `validate()` and `validate()` method."""
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Called by Trainer `test()` method."""
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader:
        """Called by Trainer `predict()` method. Use the same data as the test_dataloader."""
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=3)

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     ...
