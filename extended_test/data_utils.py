from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as transforms

from common import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DATA_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_NOISE_STD,
)


class AddGaussianNoise:
    def __init__(self, std: float = 0.0):
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std


def build_cifar10_transforms(train: bool, hardware_noise: bool, input_noise_std: float):
    transform_steps = []
    if train:
        transform_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    transform_steps.append(transforms.ToTensor())

    if hardware_noise:
        transform_steps.append(AddGaussianNoise(std=input_noise_std))

    transform_steps.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return transforms.Compose(transform_steps)


def build_cifar10_loaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    hardware_train: bool = False,
    hardware_test: bool = False,
    input_noise_std: float = DEFAULT_INPUT_NOISE_STD,
):
    trainset = torchvision.datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=build_cifar10_transforms(
            train=True,
            hardware_noise=hardware_train,
            input_noise_std=input_noise_std,
        ),
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=build_cifar10_transforms(
            train=False,
            hardware_noise=hardware_test,
            input_noise_std=input_noise_std,
        ),
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return trainloader, testloader
