import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

# Create a logger
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.INFO)

DEFAULT_IMG_SIZE = 224


class VOCSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: Path, batch_size: int = 32, val_split: float = 0.1, image_size: int = DEFAULT_IMG_SIZE
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.generator = torch.Generator().manual_seed(42)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Add any other transformations here
            ]
        )

    def setup(self, stage: str):
        logger.info(f"Stage: {stage}")
        # Transforms are already applied in the dataset initialization
        full_dataset = VOCSegmentation(
            self.data_dir, year="2012", image_set="train", download=True, transform=self.transform
        )
        # Splitting dataset into train and validation
        n_val = int(len(full_dataset) * self.val_split)
        n_train = len(full_dataset) - n_val
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=self.generator,
        )

    def prepare_data(self):
        VOCSegmentation(self.data_dir, year="2012", image_set="train", download=True, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        # For testing, load the separate test set
        test_dataset = VOCSegmentation(
            self.data_dir, year="2012", image_set="val", download=True, transform=self.transform
        )
        return DataLoader(test_dataset, batch_size=self.batch_size)
