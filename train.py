#!/usr/bin/python3

"""Example training script to fit a model on MNIST dataset."""
import os
from argparse import ArgumentParser

import lightning as L
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from datasets.mnist import MNISTDataModule
from models import AlexNet, LeNet

_PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
_BATCH_SIZE = 64 if torch.cuda.is_available() else 32
_EARLY_STOPPING_PATIENCE = 5  # epochs


def main(model_choice: str, device: int, max_epoch: int):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Select architecture
    if model_choice == "alexnet":
        model = AlexNet(in_channels=1, out_channels=10)
        # train_data = CIFAR10(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
        # train_data, val_data = random_split(train_data, [55000, 5000])
    elif model_choice == "lenet":
        model = LeNet(in_channels=1, out_channels=10)
        data_module = MNISTDataModule(data_dir=_PATH_DATASETS, batch_size=_BATCH_SIZE)

    early_stop_callback = EarlyStopping(
        monitor="loss",
        min_delta=0.00,
        patience=_EARLY_STOPPING_PATIENCE,
        verbose=True,
        mode="min",
    )
    # If your machine has GPUs, it will use the GPU Accelerator for training.
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=[device],
        strategy="auto",
        max_epochs=max_epoch,
        callbacks=[early_stop_callback],
    )

    # Train the model ⚡
    # data_module.setup(stage="fit")  # Is called by trainer.fit().
    # Call training_step + validation_step for all the epochs.
    trainer.fit(model, datamodule=data_module)
    trainer.validate(datamodule=data_module)

    # Automatically auto-loads the best weights from the previous run.
    # data_module.setup(stage="test")  # Is called by trainer.test().
    trainer.test(datamodule=data_module)
    trainer.predict(datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lenet")
    parser.add_argument("--device", default=0)
    parser.add_argument("--max-epoch", default=15)
    args = parser.parse_args()

    main(model_choice=args.model, device=args.device, max_epoch=args.max_epoch)
