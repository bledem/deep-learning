import torch.nn as nn
import torch
import torch.nn.functional as F

import lightning.pytorch as pl


class AlexNet(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Net
        # Either we create one single block with all the different layers that operates for one task in the model.

        # This implementation ignores LocalResponseNorm
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 96, stride=4, kernel_size=11),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, padding=2, kernel_size=5),  # padding same
            # Pooling keeps the same dimension but makes width and height smaller.
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # padding same
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # padding same
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1),  # padding same
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
            nn.ReLU(inplace=True),
        )

        # Classifier
        # Either we create such Sequential blocks and we have to call Relu in between the calls in the `forward()` fn.

        # Option: add DropOut between the linear layers.
        self.fc1 = nn.Linear(in_features=(256 * 6 * 6), out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=out_channels)
        self.init_bias()  # initialize bias

    def init_bias(self):
        # The advantage of using Sequential is that we can loop over blocks easily
        # with each blocks having a common meaning (net + classifier)
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper set to 1 the Conv2d layers 2nd, 4th and 5th.
        nn.init.constant_(self.net[2].bias, 1)
        nn.init.constant_(self.net[7].bias, 1)
        nn.init.constant_(self.net[9].bias, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # Flatten the tensor to on long tensor for each instance
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)