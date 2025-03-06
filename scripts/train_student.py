import torch
import os
import setup_paths
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models.mnist_student import MNISTStudent
from torch.optim import AdamW
from config import *
from torchvision.transforms import ToTensor
from PIL import Image

class SyntheticNoiseDataset(Dataset):
    def __init__(self, synthetic_dir, labeled_dir):
        self.synthetic_dir = synthetic_dir
        self.labeled_dir = labeled_dir
        self.images = sorted([f for f in os.listdir(synthetic_dir) if f.endswith(".png")], key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.labels = sorted([f for f in os.listdir(labeled_dir) if f.endswith(".txt")], key=lambda x: int(x.split("_")[-1].split(".")[0]))

    def __getitem__(self, idx):

        image_name = self.images[idx]
        label_name = self.labels[idx]

        image = Image.open(os.path.join(self.synthetic_dir, image_name)).convert("L")
        image = ToTensor()(image)

        label_path = os.path.join(self.labeled_dir, label_name)
        with open(label_path, "r") as f:
            label = float(f.read().strip())  

        label = torch.tensor(label).unsqueeze(0)  

        return image, label

    def __len__(self):
        return len(self.images)

def train_student():
    dataset = SyntheticNoiseDataset(SYNTHETIC_DIR, OUTPUT_LABELED_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MNISTStudent().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()  # Pérdida para predecir ruido

    for epoch in range(EPOCHS_STUDENT):
        epoch_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            noise = torch.randn_like(images).to(DEVICE)  # Ruido simulado
            pred_noise = model(images, noise)  # Modelo predice el ruido

            loss = loss_fn(pred_noise, labels)  # Comparar con el ruido real
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), SAVE_STUDENT_DATA_DIR)  # Guardar modelo

if __name__ == "__main__":
    train_student()
