"""In the original paper LeNet 5 the input was grayscale (32,32) squared images.
It's using 2 convolutional layers with no padding and average pooling. 
We use MaxPooling like more modern arechitecture. 
Tiny model: ~60k parameters.
As we go deeper in the model, the height and width is shrinking and the dimension is increasing.

"""
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # The activation size (number of values after passing through one layer) is getting gradually smaller and smaller.
        # The output is flatten and then used as a long input into the next dense layers.
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.fc2 = nn.Linear(in_feature=120, out_features=84)
        # "Softmax" layer = Linear + Softmax.
        self.fc3 = nn.Linear(in_features=84, out_features=out_channels)

    def forward(self, x):
        x = nn.ReLU(self.conv_layer1(x))
        x = nn.ReLU(self.conv_layer2(x))
        x = x.view(-1, 84)  # Flatten the tensor to on long tensor for each instance
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x

