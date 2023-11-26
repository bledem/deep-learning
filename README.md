# Deep Neural Network implementation in Pytorch

Informal implementation from scratch for educational purpose.

# Models

- LeNet
- AlexNet

# Dataset

- MNIST
- CalTech101
- CIFAR 10

# Library

- PyTorch
- PyTorch Lightning

## Tools

- Use `pip-compile` to update Python requirements files. Can be installed with: `python3 -m pip install pip-tools`.
- Use [`pre-commit`](https://pre-commit.com/) hooks to automate static checks:

```sh
pip install pre-commit
pre-commit install
pre-commit install --install-hooks
```

```sh

```

### Notes

Important formula to compute the size of the output tensor:
$$
\frac{W - K + 2P}{S} + 1
$$
With
- $W$ the input volume
- $K$ the filter size
- $P$ the padding size
- $S$ the stride size
