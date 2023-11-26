"""
Deep neural networks can have vanishing gradients and skip connection helps solving this issue.
1. Skip connection enables to easily learn the identity function. So adding new layers cannot hurt the performance of the training
as it can happen on deeper network w/o skip connections.
2.
The skip connection uses `g(z_{l} + a_{l-1})` with g the activation function and z the logit of the current layer,
a_{l-1} the activated logit of the previous layer. Therefore, z and a must be of the same dimension within each block where
a skip connection exists. ResNet uses same padding for solving this. Otherwise, between blocks the previous activated logits
to by multiplying with a matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.detection.base import BaseModel

_EXPANSION_SIZE = 4


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, identity_downsample=None, stride: int = 1):
        super().__init__()
        self.expansion = _EXPANSION_SIZE  # the dimension of a block is always 4 times bigger after a block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = (
            identity_downsample  # if we need to change the dimension of the input to match the output
        )

    def forward(self, x: torch.Tensor):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))
        if self.identity_downsample is not None:
            x = self.identity_downsample(x)
        x4 = self.relu(x3 + x)
        return x4


class Resnet(BaseModel):
    def __init__(self, layers: list[int], lr: float = 0.001, in_channels: int = 3, num_classes: int = 1000):
        super().__init__(learning_rate=lr, num_classes=num_classes)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3
        )  # output size 112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Resnet Layers
        in_channels, self.conv2_x = self._make_layers(
            num_residual_blocks=layers[0],  # 3 for resnet 50
            in_channels=64,
            out_channels=64,
            stride=1,
        )
        in_channels, self.conv3_x = self._make_layers(
            num_residual_blocks=layers[1],  # 4 for resnet 50
            in_channels=in_channels,
            out_channels=128,
            stride=2,
        )
        in_channels, self.conv4_x = self._make_layers(
            num_residual_blocks=layers[2],  # 4 for resnet 50
            in_channels=in_channels,
            out_channels=256,
            stride=2,
        )
        in_channels, self.conv5_x = self._make_layers(
            num_residual_blocks=layers[3],  # 4 for resnet 50
            in_channels=in_channels,
            out_channels=512,
            stride=2,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce the output to 1x1 pixel dimension
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.max_pool(self.relu(self.bn1(self.conv1(x))))
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.fc(x)
        return x

    def _make_layers(self, num_residual_blocks, in_channels, out_channels, stride):
        identity_downsample = None

        # When the block input x does not have the same dimension as the output of the block
        # Which happends when 1) The stride is not 1.
        # 2) The input channel is not equal to the output channel times 4 (which is the case after the first block of a conv)
        # Second condition is when the input channels is not equal to the output channels so we can't add the output to the input x
        if stride != 1 or in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers = [Block(in_channels, out_channels, identity_downsample, stride)]
        # The dimension of the output of the block is 4 times bigger than the input
        in_channels = out_channels * 4
        for _ in range(num_residual_blocks - 1):
            layers.append(Block(in_channels, out_channels))

        return in_channels, nn.Sequential(*layers)


def get_resnet50(in_channels: int, lr: float = 1e-4, num_classes=1000):
    return Resnet(layers=[3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes, lr=lr)


def ResNet101(in_channels: int, lr: float, num_classes=1000):
    return Resnet([3, 4, 23, 3], in_channels=in_channels, num_classes=num_classes)


def ResNet152(in_channels: int, lr: float, num_classes=1000):
    return Resnet([3, 8, 36, 3], in_channels=in_channels, num_classes=num_classes)


if __name__ == "__main__":
    model = get_resnet50(in_channels=3)
    x = torch.randn(2, 3, 224, 224)  # batch size x channels x height x width
    y = model(x).to("cuda")
    print(y, y.shape)
