import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os

from config import *

# Definir la ruta base de los datos
DATA_ROOT = os.path.join(os.path.dirname(__file__), "mnist_data")

def get_mnist_dataloaders(batch_size=BATCH_SIZE, image_size=MODEL_IMAGE_SIZE, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_dataset = MNIST(root=DATA_ROOT, train=True, download=True, transform=preprocess)
    test_dataset = MNIST(root=DATA_ROOT, train=False, download=True, transform=preprocess)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def get_mnist_prototypes():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    prototypes = torch.zeros((10, 1, 28, 28))
    counts = torch.zeros(10)
    for image, label in dataset:
        prototypes[label] += image
        counts[label] += 1
    counts[counts == 0] = 1
    prototypes /= counts.view(-1, 1, 1, 1)
    return prototypes  
