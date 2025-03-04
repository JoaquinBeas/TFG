import torch
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # FIXME: hay que arreglar los paths, esto no puede estar asi
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models.mnist_student import MNISTStudent
from torch.optim import AdamW

# Configuración
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
DATA_DIR = "src/data/labeled_synthetic"
NUM_CLASSES = 10  # MNIST has labels 0-9

class SyntheticDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

        # Filter out invalid labels
        valid_labels = []
        for file in self.labels:
            try:
                label = int(open(os.path.join(self.data_dir, file)).read().strip())
                if 0 <= label < NUM_CLASSES:
                    valid_labels.append(file)
                else:
                    print(f"⚠️ Skipping {file}: Label {label} is out of range!")
            except ValueError:
                print(f"⚠️ Skipping {file}: Invalid label format!")
        
        self.labels = valid_labels  # Keep only valid labels

    def __getitem__(self, idx):
        label_name = self.labels[idx]
        label_path = os.path.join(self.data_dir, label_name)

        # Read label from text file
        label = int(open(label_path).read().strip())

        # Create a fake "image" (random noise)
        image = torch.rand((1, 28, 28))  # 1-channel 28x28 noise

        return image, label

    def __len__(self):
        return len(self.labels)

def train_student():
    dataloader = DataLoader(SyntheticDataset(DATA_DIR), batch_size=BATCH_SIZE, shuffle=True)
    model = MNISTStudent().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_student()
