from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

SCHEDULER_STEP = 5


class BaseModel(L.LightningModule):
    def __init__(self, learning_rate: float, num_classes: int):
        super().__init__()
        self.learning_rate = learning_rate
        # Initialize metrics here, e.g., self.train_accuracy
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return optimizer

    def common_step(self, batch, batch_idx, phase):
        x, y = batch
        logit_preds = self(x)
        loss = F.cross_entropy(logit_preds, y)
        if phase == "train":
            if self.train_accuracy is not None:
                self.train_accuracy.update(torch.argmax(logit_preds, dim=1), y)
            self.log("train_acc_step", self.train_accuracy, on_epoch=True, prog_bar=True)
            # logs metrics for each training_step,
            # and the average across the epoch, to the progress bar and logger
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
            # step every N batches
            if (batch_idx + 1) % SCHEDULER_STEP == 0:
                self.lr_scheduler.step()
        elif phase == "val":
            if self.val_accuracy is not None:
                self.val_accuracy.update(torch.argmax(logit_preds, dim=1), y)
                self.log(
                    "val_acc_step",
                    self.val_accuracy,
                    on_epoch=True,
                    prog_bar=True,
                    on_step=True,
                )
            self.log("val_loss", loss)
        elif phase == "test":
            if self.test_accuracy is not None:
                self.test_accuracy.update(torch.argmax(logit_preds, dim=1), y)
                self.log("test_acc", self.test_accuracy)
            self.log("test_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        """Function called when using `trainer.fit()` with trainer a
        lightning `Trainer` instance."""
        return self.common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Function called when using `trainer.validate()` with trainer a
        lightning `Trainer` instance."""
        loss = self.common_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        """Function called when using `trainer.test()` with trainer a
        lightning `Trainer` instance."""
        return self.common_step(batch, batch_idx, "test")

    def predict_step(
        self, batch: list[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Function called when using `trainer.predict()` with trainer a
        lightning `Trainer` instance."""
        x, _ = batch
        logit_preds = self(x)
        softmax_preds = F.softmax(logit_preds, dim=1)
        return x, softmax_preds


# We can add all these functions https://pytorch-lightning.readthedocs.io/en/1.8.6/tuning/profiler_basic.html
