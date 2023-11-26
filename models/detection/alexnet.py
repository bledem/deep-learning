from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detection.base import BaseModel


class AlexNet(BaseModel):
    """The image cannot be too small otherwise the width and height will be squeezed.
    In the original setup the image size goes from: 227 -> 55 -> 27 ->
    """

    def __init__(self, in_channels: int, out_channels: int, lr: float = 2e-4):
        BaseModel.__init__(self, learning_rate=lr, num_classes=out_channels)
        # pl.LightningModule.__init__(self)
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        self.example_input_array = torch.zeros((32, 3, 227, 227), dtype=torch.float32)
        self.learning_rate = lr

        # self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        # self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        # self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)

        # Either we create one single block with all the different layers that operates for one task in the model.
        # [img_size] 227 -> (conv) 55 -> (maxpool) -> 27
        # it is a large image width/height shrink from 227 to 55 mostly due to a large stride (s=4).
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, stride=4, kernel_size=11),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        )
        # [img_size] 27 -> (conv) 27 -> (maxpool) -> 13
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, padding=2, kernel_size=5),  # padding same
            # Pooling keeps the same dimension but makes width and height smaller.
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        )
        # All the next convs are added to one single Sequential block for
        # simplifying its initiatlization in `init_bias()`.
        self.net = nn.Sequential(
            # [img_size] -> 13 -> .... -> 6 on last Max Pool
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # padding same
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # padding same
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # padding same
            nn.MaxPool2d(kernel_size=3, stride=2),  # (batch_size x 256 x 6 x 6)
            nn.ReLU(),
        )

        # Classifier
        # Either we create such Sequential blocks and we have to call Relu in between the calls in the `forward()` fn.
        # Option: add DropOut between the linear layers.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=out_channels),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        # The advantage of using Sequential is that we can loop over blocks easily
        # with each blocks having a common meaning (net + classifier)
        for layer in self.net + self.conv1 + self.conv2:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper set to 1 the Conv2d layers 2nd, 4th and 5th.
        nn.init.constant_(self.conv2[0].bias, 1)
        nn.init.constant_(self.net[2].bias, 1)
        nn.init.constant_(self.net[4].bias, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # Flatten the tensor to on long tensor for each instance
        x = self.classifier(x)
        return x

    # def training_step(
    #     self,
    #     batch: list[torch.Tensor, torch.Tensor],
    #     batch_idx: int,
    # ) -> torch.Tensor:
    #     """Function called when using `trainer.fit()` with trainer a
    #     lightning `Trainer` instance."""
    #     x, y = batch
    #     logit_preds = self(x)
    #     loss = F.cross_entropy(logit_preds, y)
    #     self.train_accuracy.update(torch.argmax(logit_preds, dim=1), y)
    #     self.log("train_acc_step", self.train_accuracy, on_step=True, on_epoch=True, logger=True)
    #     # logs metrics for each training_step,
    #     # and the average across the epoch, to the progress bar and logger
    #     self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
    #     # step every N batches
    #     if (batch_idx + 1) % _SCHEDULER_STEP == 0:
    #         self.lr_scheduler.step()
    #     return loss

    # def validation_step(
    #     self,
    #     batch: list[torch.Tensor, torch.Tensor],
    #     batch_idx: int,
    #     verbose: bool = True,
    # ) -> torch.Tensor:
    #     """Function called when using `trainer.validate()` with trainer a
    #     lightning `Trainer` instance."""
    #     x, y = batch
    #     logit_preds = self(x)
    #     loss = F.cross_entropy(logit_preds, y)
    #     self.val_accuracy.update(torch.argmax(logit_preds, dim=1), y)
    #     self.log("val_loss", loss)
    #     self.log("val_acc_step", self.val_accuracy, on_epoch=True)
    #     return loss

    # def test_step(
    #     self,
    #     batch: list[torch.Tensor, torch.Tensor],
    #     batch_idx: int,
    # ):
    #     """Function called when using `trainer.test()` with trainer a
    #     lightning `Trainer` instance."""
    #     x, y = batch
    #     logit_preds = self(x)
    #     loss = F.cross_entropy(logit_preds, y)
    #     self.test_accuracy.update(torch.argmax(logit_preds, dim=1), y)
    #     self.log_dict({"test_loss": loss, "test_acc": self.test_accuracy})

    # def predict_step(
    #     self, batch: list[torch.Tensor, torch.Tensor], batch_idx: int
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Function called when using `trainer.predict()` with trainer a
    #     lightning `Trainer` instance."""
    #     x, _ = batch
    #     logit_preds = self(x)
    #     softmax_preds = F.softmax(logit_preds, dim=1)
    #     return x, softmax_preds

    # def configure_optimizers(self) -> torch.optim.Adam:
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     self.lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #     return optimizer
