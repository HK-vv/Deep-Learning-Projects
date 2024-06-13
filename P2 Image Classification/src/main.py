import torch
import torchvision
from torch.utils.data import dataloader

EPOCH=100
BATCH_SIZE=100

# Data Preparation
train_set=torchvision.datasets.CIFAR10(download=True, root='../data', train=True, 
									   transform=torchvision.transforms.ToTensor())

test_set=torchvision.datasets.CIFAR10(download=True, root='../data', train=False, 
									  transform=torchvision.transforms.ToTensor())

train_loader=dataloader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader=dataloader(test_set, batch_size=BATCH_SIZE, )



