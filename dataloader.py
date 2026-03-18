"""MNIST dataloaders for the main pipeline with noisy-input training."""

import torchvision
import torchvision.transforms as transforms
import torch
from args import *

image_size=args.image_size
INPUT_NOISE_STD = max(float(args.link_calibrated_noise), 0.0)


class AddGaussianNoise(object):
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, tensor):
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


trainset = torchvision.datasets.MNIST(root=".//data//",
                            transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((image_size,image_size)),
            torchvision.transforms.ToTensor(),
            # Train under noisy inputs to mimic hardware-facing input perturbations.
            AddGaussianNoise(std=INPUT_NOISE_STD),
        ]
    ),
                            train=True,
                            download=True)
testset = torchvision.datasets.MNIST(root=".//data//",
                           transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize((image_size,image_size)), torchvision.transforms.ToTensor()]
    ),
                           train=False,
                           download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
