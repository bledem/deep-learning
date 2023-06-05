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

import torch.nn as nn


class Resnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        # TODO
        pass
