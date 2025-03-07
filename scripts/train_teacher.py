import setup_paths
import sys
import torch
import os
import math
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import AdamW
from torchvision.utils import save_image
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
from models.mnist_teacher import MNISTDiffusion
from utils.exponential_moving_avg import ExponentialMovingAverage
from config import *
from data.data_loader import get_mnist_dataloaders

# Entrenamiento del modelo
def train_model(train_loader, epochs=EPOCHS_TEACHER, _lr=LEARNING_RATE, device="cuda"):
    model = MNISTDiffusion(timesteps=TIMESTEPS, image_size=MODEL_IMAGE_SIZE, in_channels=MODEL_IN_CHANNELS, base_dim=MODEL_BASE_DIM, dim_mults=MODEL_DIM_MULTS).to(device)
    adjust = 1* BATCH_SIZE * MODEL_EMA_STEPS / EPOCHS_TEACHER
    alpha = 1.0 - MODEL_EMA_DECAY
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)    
    optimizer = AdamW(model.parameters(), lr=_lr)
    scheduler = OneCycleLR(optimizer, _lr, total_steps=epochs * len(train_loader), pct_start=0.25, anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')
    os.makedirs(SAVE_TEACHER_DATA_DIR, exist_ok=True)
    if CKTP:
        cktp=torch.load(CKTP)
        model_ema.load_state_dict(cktp["model_ema"]) #modelo suavizado
        model.load_state_dict(cktp["model"])         #modelo normal       
    global_steps = 0
    for epoch in range(epochs):
        model.train()
        for step, (images, _) in enumerate(train_loader):
            noise = torch.randn_like(images).to(device)
            images = images.to(device)
            pred_noise = model(images, noise)
            loss = loss_fn(pred_noise, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()   
            if global_steps % MODEL_EMA_STEPS == 0:
                model_ema.update_parameters(model)
            global_steps += 1   
            if step % LOG_FREQ == 0:
                print(f"Epoch[{epoch+1}/{epochs}], Step[{step}/{len(train_loader)}], Loss: {loss.item():.5f}, LR: {scheduler.get_last_lr()[0]:.5f}")
        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()} 
        model_ema.eval()
        samples = model_ema.module.sampling(N_SAMPLES_TRAIN, clipped_reverse_diffusion=True, device=device)
        save_image(samples, SAVE_TEACHER_DATA_DIR+f"/epoch_{epoch+1}.png", nrow=int(math.sqrt(N_SAMPLES_TRAIN)), normalize=True)
        # torch.save(ckpt, f"src/data/train/teacher_epochs/epoch_{epoch+1}.pt") Guardar cada modelo
    torch.save(ckpt, MODEL_PATH) # Guardar solo el último modelo 
    return model

if __name__ == "__main__":
    train_loader, test_loader = get_mnist_dataloaders()
    train_model(train_loader)
