import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # Batch norm is a linear operation that adds a bias,
            # it in uncencessary to have two bias before the non linear activation.
            nn.Conv2d(
                in_channels=in_channels,  # Dimension of one filter sliding across the input.
                out_channels=out_channels,  # Number of filters.
                kernel_size=3,
                # bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels=out_channels,
                in_channels=out_channels,
                kernel_size=3,
                # bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            DoubleConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
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
        # Because we divide the image height and width per 2 at each down layer, the image must be divisible by 2^len(down_layers).
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        inputs_features = [
            in_channels,
            64,
            128,
            256,
            512,
        ]  # At each new block the number of features channels is doubled.
        for i in range(len(inputs_features) - 1):
            self.downs.append(DoubleConv(inputs_features[i], inputs_features[i + 1]))
        self.bottleneck_conv = DoubleConv(inputs_features[-1], inputs_features[-1] * 2)
        self.ups.append(DoubleConv(512, 1024))
        inv_feat = inputs_features[::-1]
        for i in range(len(inputs_features) - 1):
            self.ups.append(UpDoubleConv(inv_feat[i], inv_feat[i + 1]))
        self.last_conv = nn.Conv2d(inputs_features[1], self.num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down_conv in self.downs:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck_conv(x)
        for up_conv in self.ups:
            x = up_conv(x)
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
        return self.last_conv(x)


if __name__ == "__main__":
    model = UNet()
    x_test = torch.randn(1, 3, 572, 572)
    y = model(x_test)
    print(y.shape)
    print(model)
