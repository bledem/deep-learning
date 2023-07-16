"""In the original paper LeNet-5, the input are grayscale (32,32) squared images.
The model is composed of 2 convolutional layers with no padding and uses average pooling.
In this implementation, we use MaxPooling like in more modern architecture.
Tiny model: ~60k parameters.
As we go deeper in the model, the height and width is shrinking and the dimension is increasing.

PyTorch reference: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class LeNet(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
        - in_channels: One for grayscale input image, 3 for RGB input image.
        - out_channels: Number of classes of the classifier. 10 for MNIST.
        """
        super().__init__()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        # [img_size] 32 -> conv -> 32 -> (max_pool) -> 16
        # with 6 output activation maps
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=5,
                stride=1,
                # Either resize (28x28) MNIST images to (32x32) or pad the imput to be 32x32
                # padding=2,
            ),
            nn.MaxPool2d(kernel_size=2),
        )
        # [img_size] 16 -> (conv) -> 10 -> (max pool) 5
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # The activation size (number of values after passing through one layer) is getting gradually smaller and smaller.
        # The output is flatten and then used as a long input into the next dense layers.
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)  # 5 from the image dimension
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # "Softmax" layer = Linear + Softmax.
        self.fc3 = nn.Linear(in_features=84, out_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_layer1(x))
        x = F.relu(self.conv_layer2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    ###############################
    # --- For Pytorch Lightning ---
    ###############################
    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def validation_step(
        self,
        batch: list[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Function called when using `trainer.validate()` with trainer a
        lightning `Trainer` instance."""
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        self.log("loss", loss)
        self.accuracy(preds, y)
        self.log("acc_step", self.accuracy, on_epoch=True)
        return loss

    def training_step(
        self,
        batch: list[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Function called when using `trainer.fit()` with trainer a
        lightning `Trainer` instance."""
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        self.accuracy(preds, y)
        self.log("train_acc_step", self.accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(
        self,
        batch: list[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        """Function called when using `trainer.test()` with trainer a
        lightning `Trainer` instance."""
        test_loss = self.validation_step(batch, batch_idx)
        self.log_dict({"test_loss": test_loss})

    def predict_step(
        self,
        batch: list[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """Function called when using `trainer.predict()` with trainer a
        lightning `Trainer` instance."""
        x, _ = batch
        return self(x)
