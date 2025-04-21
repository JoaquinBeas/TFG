import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, ImageFolder
from torchvision import transforms

from src.utils.config import BATCH_SIZE, IMAGE_SIZE, MNIST_DATA_LOADERS_DIR, NUM_WORKERS, SAVE_SYNTHETIC_DATASET_DIR

def get_mnist_dataloaders(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=NUM_WORKERS):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_dataset = MNIST(root=MNIST_DATA_LOADERS_DIR, train=True, download=True, transform=preprocess)
    test_dataset = MNIST(root=MNIST_DATA_LOADERS_DIR, train=False, download=True, transform=preprocess)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def get_synthetic_mnist_dataloaders(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=NUM_WORKERS, synthetic_data_dir=None):
    if synthetic_data_dir is None:
        synthetic_data_dir = SAVE_SYNTHETIC_DATASET_DIR

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = ImageFolder(root=synthetic_data_dir, transform=transform)
    total_len = len(dataset)
    train_size = int(0.85 * total_len)
    test_size = total_len - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_mnist_prototypes():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root=MNIST_DATA_LOADERS_DIR, train=True, download=True, transform=transform)
    prototypes = torch.zeros((10, 1, 28, 28))
    counts = torch.zeros(10)
    for image, label in dataset:
        prototypes[label] += image
        counts[label] += 1
    counts[counts == 0] = 1
    prototypes /= counts.view(-1, 1, 1, 1)
    return prototypes
