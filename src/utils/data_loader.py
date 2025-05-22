import gzip
import os
from pathlib import Path
import struct
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
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

def get_synthetic_mnist_dataloaders(
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    num_workers=NUM_WORKERS,
    synthetic_data_dir=None
):
    if synthetic_data_dir is None:
        synthetic_data_dir = SAVE_SYNTHETIC_DATASET_DIR

    # 1) Load IDX-gzip files directly
    img_path = os.path.join(synthetic_data_dir, "train-images-idx3-ubyte.gz")
    with gzip.open(img_path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        buf = f.read(n * h * w)
    images = np.frombuffer(buf, dtype=np.uint8).reshape(n, 1, h, w)

    lbl_path = os.path.join(synthetic_data_dir, "train-labels-idx1-ubyte.gz")
    with gzip.open(lbl_path, "rb") as f:
        magic, n2 = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(n2), dtype=np.uint8)
    assert n == n2, "Mismatch between image and label count"

    # 2) Convert to torch tensors and normalize to [-1,1]
    imgs_t = torch.from_numpy(images).float().div(255.0).mul(2).sub(1)  # [N,1,H,W] in [-1,1]
    lbls_t = torch.from_numpy(labels).long()

    # 3) Build a TensorDataset
    full_ds = TensorDataset(imgs_t, lbls_t)

    # 4) Split into train/test
    total = len(full_ds)
    train_len = int(0.85 * total)
    test_len  = total - train_len
    train_ds, test_ds = random_split(full_ds, [train_len, test_len])

    # 5) DataLoaders
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader

def get_all_synthetic_dataloaders(
    base_dir: str = SAVE_SYNTHETIC_DATASET_DIR,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMAGE_SIZE,
    num_workers: int = NUM_WORKERS,
):
    """
    Devuelve un dict {nombre_dataset: (train_loader, val_loader)} para cada
    subcarpeta que contenga los ficheros IDX necesarios.
    """
    loaders = {}
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"No se encontró la carpeta {base_dir}")

    for sub in base.iterdir():
        if (
            sub.is_dir()
            and (sub / "train-images-idx3-ubyte.gz").exists()
            and (sub / "train-labels-idx1-ubyte.gz").exists()
        ):
            tl, vl = get_synthetic_mnist_dataloaders(
                batch_size=batch_size,
                image_size=img_size,
                num_workers=num_workers,
                synthetic_data_dir=str(sub),
            )
            loaders[sub.name] = (tl, vl)

    return loaders

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
