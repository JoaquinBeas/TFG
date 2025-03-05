import torch
import sys
import os
import setup_paths
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models.mnist_student import MNISTStudent
from torch.optim import AdamW
from config import *

class SyntheticDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

        # Filter out invalid labels # LAS labels se salen del rango TODO: ARREGLAR ESTO
        valid_labels = []
        for file in self.labels:
            try:
                label = int(open(os.path.join(self.data_dir, file)).read().strip())
                if 0 <= label < 10:
                    valid_labels.append(file)
                else:
                    print(f"⚠️ Skipping {file}: Label {label} is out of range!")
            except ValueError:
                print(f"⚠️ Skipping {file}: Invalid label format!")
        self.labels = valid_labels

    def __getitem__(self, idx):
        label_name = self.labels[idx]
        label_path = os.path.join(self.data_dir, label_name)
        label = int(open(label_path).read().strip())
        image = torch.rand((1, 28, 28))
        return image, label

    def __len__(self):
        return len(self.labels)

def train_student():
    dataloader = DataLoader(SyntheticDataset(DATA_DIR), batch_size=BATCH_SIZE, shuffle=True)
    model = MNISTStudent().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS_STUDENT):
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_student()
