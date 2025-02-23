import torch
import numpy as np
import random
import os
import math
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.model import MNISTDiffusion
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
from utils.exponential_moving_avg import ExponentialMovingAverage
import argparse

# Constants
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
MODEL_BASE_DIM = 64
TIMESTEPS = 1000
MODEL_EMA_STEPS = 10
MODEL_EMA_DECAY = 0.995
LOG_FREQ = 10
N_SAMPLES = 36
USE_CPU = False
CKTP = ''

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load MNIST dataset
def create_mnist_dataloaders(batch_size=BATCH_SIZE, image_size=28, num_workers=4):
    set_seed(42)
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
    test_dataset = MNIST(root="./mnist_data", train=False, download=True, transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader

# Train the MNISTDiffusion model
def train_model(train_loader, epochs=EPOCHS, _lr=LEARNING_RATE, device="cuda"):
    model = MNISTDiffusion(timesteps=TIMESTEPS, image_size=28, in_channels=1, base_dim=MODEL_BASE_DIM, dim_mults=[2, 4]).to(device)
    adjust = 1* BATCH_SIZE * MODEL_EMA_STEPS / EPOCHS
    alpha = 1.0 - MODEL_EMA_DECAY
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)    
    optimizer = AdamW(model.parameters(), lr=_lr)
    scheduler = OneCycleLR(optimizer, _lr, total_steps=epochs * len(train_loader), pct_start=0.25, anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')
    os.makedirs("results", exist_ok=True)

    #load checkpoint
    if CKTP:
        cktp=torch.load(CKTP)
        model_ema.load_state_dict(cktp["model_ema"])
        model.load_state_dict(cktp["model"])
        
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
        samples = model_ema.module.sampling(N_SAMPLES, clipped_reverse_diffusion=True, device=device)
        save_image(samples, f"results/epoch_{epoch+1}.png", nrow=int(math.sqrt(N_SAMPLES)), normalize=True)
        
        torch.save(ckpt, f"results/epoch_{epoch+1}.pt")

    return model

# Obtain predictions from the model
def obtain_predictions(model, test_loader, device="cuda"):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            noise = torch.randn_like(images).to(device)
            images = images.to(device)
            pred = model(images, noise)
            predictions.append(pred.cpu())
            targets.append(labels.cpu())
    return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)

# Evaluate model accuracy
def eval_model(predictions, targets):
    diff = (predictions.argmax(dim=1) == targets).float()
    accuracy = diff.mean().item()
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Main execution
def main():
    device = "cpu" if USE_CPU else "cuda"
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    train_loader, test_loader = create_mnist_dataloaders()
    
    # Train the model
    print("Training model...")
    trained_model = train_model(train_loader, device=device)
    
    # Obtain predictions and evaluate
    print("Evaluating model...")
    predictions, targets = obtain_predictions(trained_model, test_loader, device=device)
    eval_model(predictions, targets)


if __name__ == "__main__":
    main()
