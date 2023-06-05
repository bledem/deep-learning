"""Implementation of VGG16 from scratch in Pytorch.
Conv 3x3, stride is 1, padding is same. Max pool is 2x2, stride is 2. 
16 refers to the 16 layers and then ~138M parameters. Which is quite big. 
"""

import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, in_channels: int=3, out_channels:int):
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, stride=1, kernel_size=3, padding=1), # padding same
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1), # (b, 224, 224, 64)
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2), # (b, 112, 112, 64)

            nn.Conv2d(
                in_channels=64, out_channels=128, padding=1, kernel_size=3, stride=1
            ),  
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, padding=1, kernel_size=3, stride=1
            ),  # (b, 112, 112, 128)
            # Pooling keeps the same dimension but makes width and height smaller.
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (b, 56, 56, 128)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (b, 56, 56, 256)
            nn.MaxPool2d(kernel_size=2, stride=2), # (b, 28, 28, 256)
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  
            nn.MaxPool2d(kernel_size=2, stride=2), # (b, 14, 14, 512)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  
            nn.MaxPool2d(kernel_size=2, stride=2), # (b, 7, 7, 512)
            nn.ReLU(inplace=True),)
        
        self.classifier = nn.Sequential(nn.Linear(512*7*7,  4096),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4096, out_features=out_channels)
                                            )
        self.weight_init()

    def weight_init(self): 
        pass

    def forward(self, x)-> torch.Tensor:
        x = self.net(x)
        x = x.view(-1, 512*7*7)
        x = self.classifier(x)
        return x

           