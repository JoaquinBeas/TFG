import sys
import torch
import os
import math
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import AdamW
from torchvision.utils import save_image
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn

from src.models.mnist_teacher import MNISTDiffusion
from src.utils.exponential_moving_avg import ExponentialMovingAverage
from src.config import *
from src.utils.data_loader import get_mnist_dataloaders
from src.utils.training_plot import plot_training_curves

# Entrenamiento del modelo
def train_model(train_loader, epochs=EPOCHS_TEACHER, _lr=LEARNING_RATE, device="cuda", patience=8, min_delta=1e-6):
    full_dataset = train_loader.dataset
    total_len = len(full_dataset)
    val_len = int(0.15 * total_len)
    train_len = total_len - val_len
    train_subset, val_subset = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MNISTDiffusion(timesteps=TIMESTEPS, image_size=MODEL_IMAGE_SIZE, in_channels=MODEL_IN_CHANNELS, base_dim=MODEL_BASE_DIM, dim_mults=MODEL_DIM_MULTS).to(device)
    adjust = 1* BATCH_SIZE * MODEL_EMA_STEPS / EPOCHS_TEACHER
    alpha = 1.0 - MODEL_EMA_DECAY
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)    
    optimizer = AdamW(model.parameters(), lr=_lr)
    scheduler = OneCycleLR(optimizer, _lr, total_steps=epochs * len(train_loader), pct_start=0.25, anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')
    os.makedirs(SAVE_TEACHER_IMAGES_DIR, exist_ok=True)
    os.makedirs(SAVE_MODELS_TEACHER_DIR, exist_ok=True)
    if CKTP:
        cktp=torch.load(CKTP)
        model_ema.load_state_dict(cktp["model_ema"]) #modelo suavizado
        model.load_state_dict(cktp["model"])         #modelo normal       
    global_steps = 0    
    train_losses = []
    val_losses = []
    
    # Variables para early stopping
    best_val_loss = float('inf')
    counter = 0
    best_model = None
    best_model_ema = None
    stopped_epoch = epochs

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for step, (images, _) in enumerate(train_loader):
            noise = torch.randn_like(images).to(device)
            images = images.to(device)
            pred_noise = model(images, noise)
            loss = loss_fn(pred_noise, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()   
            running_train_loss += loss.item()
            if global_steps % MODEL_EMA_STEPS == 0:
                model_ema.update_parameters(model)
            global_steps += 1   
            if step % LOG_FREQ == 0:
                print(f"Epoch[{epoch+1}/{epochs}], Step[{step}/{len(train_loader)}], Loss: {loss.item():.8f}, LR: {scheduler.get_last_lr()[0]:.8f}")
        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()} 
        model_ema.eval()
        
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                noise = torch.randn_like(images).to(device)
                pred_noise = model(images, noise)
                val_loss = loss_fn(pred_noise, noise)
                total_val_loss += val_loss.item()
                val_steps += 1
        
        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else 0
        val_losses.append(avg_val_loss)
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        samples = model_ema.module.sampling(N_SAMPLES_TRAIN, clipped_reverse_diffusion=True, device=device)
        save_image(samples, os.path.join(SAVE_TEACHER_IMAGES_DIR, f"epoch_{epoch+1}.png"),
           nrow=int(math.sqrt(N_SAMPLES_TRAIN)), normalize=True)
        
        # Guardar checkpoint del modelo actual
        torch.save(ckpt, os.path.join(SAVE_MODELS_TEACHER_DIR, f"model_teacher_{epoch+1}.pt"))
        
        # Comprobar early stopping
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")
        
        # Si la pérdida de validación ha mejorado más que min_delta
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            counter = 0
            # Guardar el mejor modelo
            best_model = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            best_model_ema = {key: value.cpu().clone() for key, value in model_ema.state_dict().items()}
            # Guardar el mejor modelo en un archivo separado
            best_ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict(), "epoch": epoch+1}
            torch.save(best_ckpt, os.path.join(SAVE_MODELS_TEACHER_DIR, "best_model_teacher.pt"))
            print(f"Nuevo mejor modelo guardado (Val Loss: {best_val_loss:.8f})")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping activado en la época {epoch+1}")
                stopped_epoch = epoch + 1
                break
    
    # Si se detuvo temprano, cargar el mejor modelo para la generación final de imágenes
    if counter >= patience and best_model is not None and best_model_ema is not None:
        model.load_state_dict(best_model)
        model_ema.load_state_dict(best_model_ema)
        print(f"Modelo restaurado a la mejor época con pérdida de validación: {best_val_loss:.8f}")
    torch.save(best_ckpt, os.path.join(LAST_TEACHER_CKPT))
    # Generar imágenes con el mejor modelo
    model_ema.eval()
    final_samples = model_ema.module.sampling(N_SAMPLES_TRAIN, clipped_reverse_diffusion=True, device=device)
    save_image(final_samples, os.path.join(SAVE_TEACHER_IMAGES_DIR, "final_samples.png"),
       nrow=int(math.sqrt(N_SAMPLES_TRAIN)), normalize=True)

    curve_path = os.path.join(SAVE_TEACHER_IMAGES_DIR, "training_curve.png")
    plot_training_curves(train_losses, val_losses, curve_path)
    print(f"Gráfica de entrenamiento guardada en: {curve_path}")

    return model

if __name__ == "__main__":
    train_loader, test_loader = get_mnist_dataloaders()
    train_model(train_loader)