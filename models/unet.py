import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(self).__init__()
        self.conv = nn.Sequential(
            # Batch norm is a linear operation that adds a bias,
            # it in uncencessary to have two bias before the non linear activation.
            nn.Conv2d(
                out_channels=out_channels,  # Number of filters.
                in_channels=in_channels,  # Dimension of one filter sliding across the input.
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels=out_channels,
                in_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(in_place=True),
        )


class UNet(nn.Module):
    def __init__(
        self, num_classes: int = 2, in_channels: int = 3, features=[64, 126, 256, 512]
    ):
        super().__init__()
        self.num_classes = num_classes
        # List of modules such as the element it contains are registered and visible to other module methods.
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Hyper-params 1. Kernel size is the filter size over which we compute the max.
        # Hyper-params 2. Stride is how much we move the window to compute a new max.
        # Intuition: we preserve the high values (which corresponds to detected values for a given filters)
        # among a patch of neighbors.
        # More at: https://www.youtube.com/watch?v=8oOgPUO-TBY&ab_channel=DeepLearningAI
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
