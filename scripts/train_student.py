import torch
from torch.utils.data import DataLoader, Dataset
import os
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

# Dataset personalizado
class SyntheticDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = [f for f in os.listdir(data_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.data_dir, image_name)
        label_path = image_path.replace(".png", ".txt")

        image = Image.open(image_path).convert("L")
        image = ToTensor()(image)

        with open(label_path, "r") as f:
            label = int(f.read().strip())

        return image, label

# Entrenamiento del modelo copia
def train_student():
    dataset = SyntheticDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MNISTStudent().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("Entrenamiento del modelo copia finalizado.")

if __name__ == "__main__":
    train_student()
