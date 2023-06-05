from argparse import ArgumentParser
import torch
import os
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import lightning as L

from models.detection.alexnet import AlexNet

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


def main(model_choice: str):
    if model_choice == "alexnet":
        model = AlexNet(in_channels=1, out_channels=10)
    train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    trainer = L.Trainer(accelerator="auto", devices=1, max_epoch=3)
    # Train the model ⚡
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args.model)
