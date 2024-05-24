import torchvision
import torchvision.transforms as transforms
import torch
from args import *

image_size=args.image_size
trainset = torchvision.datasets.MNIST(root=".//data//",
                            transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize((image_size,image_size)), torchvision.transforms.ToTensor()]
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
