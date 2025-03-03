import torch
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn as nn
from models.mnist_student import MNISTStudent
from torch.optim import AdamW

# Configuración
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
DATA_DIR = "data/labeled_synthetic"

class SyntheticDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = [f for f in os.listdir(data_dir) if f.endswith(".png")]

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = Image.open(os.path.join(self.data_dir, image_name)).convert("L")
        label = int(open(os.path.join(self.data_dir, image_name.replace(".png", ".txt"))).read().strip())
        return ToTensor()(image), label

    def __len__(self):
        return len(self.images)

def train_student():
    dataloader = DataLoader(SyntheticDataset(DATA_DIR), batch_size=BATCH_SIZE, shuffle=True)
    model = MNISTStudent().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_student()
