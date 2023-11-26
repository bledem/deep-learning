#!/usr/bin/python3

"""Example training script to fit a model on MNIST dataset."""
from __future__ import annotations  # Enable PEP 563 for Python 3.7

import os
import random
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

import lightning as L
import torch
import torchvision
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from datasets.caltech101 import Caltech101
from datasets.mnist import MNISTDataModule
from datasets.pascalvoc import VOCSegmentationDataModule
from models import AlexNet, LeNet, UNet, get_resnet50
from randomness_utils import set_all_seeds, set_deterministic

_PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
_BATCH_SIZE = 64 if torch.cuda.is_available() else 32
_EARLY_STOPPING_PATIENCE = 4  # epochs
_RANDOM_SEED = 42

set_all_seeds(_RANDOM_SEED)


class ModelName(Enum):
    ALEXNET = "alexnet"
    LENET = "lenet"
    RESNET = "resnet"


def save_results(
    img_tensors: list[torch.Tensor], output_tensors: list[torch.Tensor], out_dir: Path, max_number_of_imgs: int = 10
):
    """Save test results as images in the provided output directory.
    Args:
        img_tensors: List of the tensors containing the input images.
        output_tensors: List of softmax activation from the trained model.
        out_dir: Path to output directory.
        max_number_of_imgs: Maximum number of images to output from the provided images. The images will be selected randomly.
    """
    selected_img_indices = random.sample(range(len(img_tensors)), min(max_number_of_imgs, len(img_tensors)))
    for img_indice in selected_img_indices:
        # Take the first instance of the batch (index 0)
        img_filepath = out_dir / f"{img_indice}_predicted_{torch.argmax(output_tensors[img_indice], dim=1)[0]}.png"
        torchvision.utils.save_image(img_tensors[img_indice][0], fp=img_filepath)


def main(
    model_choice: ModelName,
    device: int,
    max_epoch: int,
    out_dir: Path | None,
    early_stopping: bool | None,
):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if out_dir is None:
        out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Select architecture
    if model_choice == ModelName.ALEXNET.value:
        model = AlexNet(in_channels=3, out_channels=101, lr=2e-3)
        data_module = Caltech101(data_dir=_PATH_DATASETS, batch_size=_BATCH_SIZE)
    elif model_choice == ModelName.LENET.value:
        model = LeNet(in_channels=1, out_channels=10)
        data_module = MNISTDataModule(data_dir=_PATH_DATASETS, batch_size=_BATCH_SIZE)
    elif model_choice == ModelName.RESNET.value:
        # model = get_resnet50(in_channels=3, num_classes=101, lr=2e-4)
        # data_module = Caltech101(data_dir=_PATH_DATASETS, batch_size=_BATCH_SIZE, image_size=224)
        model = get_resnet50(in_channels=1, num_classes=10, lr=2e-4)
        data_module = MNISTDataModule(data_dir=_PATH_DATASETS, batch_size=_BATCH_SIZE)

    elif model_choice == "unet":
        model = UNet(in_channels=3, num_classes=21)
        data_module = VOCSegmentationDataModule(data_dir=_PATH_DATASETS, batch_size=_BATCH_SIZE, image_size=224)
    else:
        raise NotImplementedError(f"Model {model_choice} is not implemented.")
    callbacks = (
        [
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=_EARLY_STOPPING_PATIENCE,
                verbose=True,
                mode="min",
            )
        ]
        if early_stopping
        else []
    )

    # If your machine has GPUs, it will use the GPU Accelerator for training.
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=[device],
        strategy="auto",
        max_epochs=max_epoch,
        callbacks=callbacks,
        default_root_dir=out_dir,
    )

    # Train the model ⚡
    # data_module.setup(stage="fit")  # Is called by trainer.fit().
    # Call training_step + validation_step for all the epochs.
    trainer.fit(model, datamodule=data_module)
    # Validate
    trainer.validate(datamodule=data_module)

    # Automatically auto-loads the best weights from the previous run.
    # data_module.setup(stage="test")  # Is called by trainer.test().
    # The checkpoint path is logged on the terminal.
    trainer.test(datamodule=data_module)

    # Run the prediction on the test set and save a subset of the resulting prediction along with the
    # original image.

    output_preds = trainer.predict(datamodule=data_module, ckpt_path="best")
    img_tensors, softmax_preds = zip(*output_preds)
    out_dir_imgs = out_dir / "test_images"
    out_dir_imgs.mkdir(exist_ok=True, parents=True)
    save_results(
        img_tensors=img_tensors,
        output_tensors=softmax_preds,
        out_dir=out_dir_imgs,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet", type=str, help="Provide an implemented model.")
    parser.add_argument("--device", default=0, type=int, help="Select a CUDA device.")
    parser.add_argument("--max-epoch", default=20, type=int, help="Max number of epochs.")
    parser.add_argument("--out-dir", type=Path, help="Path to output directory")
    parser.add_argument(
        "--early-stopping", action="store_true", help="If True, stops the training if validation loss stops decreasing."
    )

    args = parser.parse_args()

    main(
        model_choice=args.model,
        device=args.device,
        max_epoch=args.max_epoch,
        out_dir=args.out_dir,
        early_stopping=args.early_stopping,
    )
