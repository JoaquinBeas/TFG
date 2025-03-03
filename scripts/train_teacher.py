import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import os
from models.mnist_teacher import MNISTDiffusion
from utils.exponential_moving_avg import ExponentialMovingAverage

# Configuración
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
TIMESTEPS = 1000
MODEL_EMA_STEPS = 10
MODEL_EMA_DECAY = 0.995
LOG_FREQ = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "data/train_epochs"

# Cargar MNIST
def create_dataloaders(batch_size=BATCH_SIZE, image_size=28):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader

# Entrenamiento del modelo
def train_model():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    train_loader = create_dataloaders()
    model = MNISTDiffusion(timesteps=TIMESTEPS, image_size=28, in_channels=1).to(DEVICE)
    model_ema = ExponentialMovingAverage(model, device=DEVICE, decay=MODEL_EMA_DECAY)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = OneCycleLR(optimizer, LEARNING_RATE, total_steps=EPOCHS * len(train_loader), pct_start=0.25)
    loss_fn = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        for step, (images, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            noise = torch.randn_like(images).to(DEVICE)
            pred_noise = model(images, noise)
            
            loss = loss_fn(pred_noise, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            if step % LOG_FREQ == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.5f}")

        # Guardar modelo cada época
        torch.save(model.state_dict(), f"{SAVE_DIR}/epoch_{epoch+1}.pth")

    print("Entrenamiento del modelo maestro finalizado.")

if __name__ == "__main__":
    train_model()
