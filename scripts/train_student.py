import torch
import os
import setup_paths
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models.mnist_student import MNISTStudent
from torch.optim import AdamW
from config import *
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from torchvision.utils import save_image
from torch.optim.lr_scheduler import OneCycleLR
import math
from torch.utils.data import ConcatDataset
from utils.exponential_moving_avg import ExponentialMovingAverage

class SyntheticNoiseDataset(Dataset):
    def __init__(self, synthetic_dir, labeled_dir):
        self.synthetic_dir = synthetic_dir
        self.labeled_dir = labeled_dir
        self.images = sorted(
            [f for f in os.listdir(synthetic_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        self.labels = sorted(
            [f for f in os.listdir(labeled_dir) if f.endswith(".txt")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        self.transform = Compose([
            Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)),  # Forzar tamaño 28x28
            ToTensor()
        ])

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label_name = self.labels[idx]
        image = Image.open(os.path.join(self.synthetic_dir, image_name)).convert("L")
        image = self.transform(image)
        label_path = os.path.join(self.labeled_dir, label_name)
        with open(label_path, "r") as f:
            label = float(f.read().strip())
        label = torch.tensor(label).unsqueeze(0)
        return image, label

    def __len__(self):
        return len(self.images)

def train_student(epochs=EPOCHS_STUDENT, lr=LEARNING_RATE, device=DEVICE):
    # Dataset y DataLoader
    dataset = SyntheticNoiseDataset(SYNTHETIC_DIR, OUTPUT_LABELED_DIR)    
    synthetic_dataset_repeated = ConcatDataset([dataset] * 30) # Esque el dataset era muy pequeño
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader = DataLoader(synthetic_dataset_repeated, batch_size=BATCH_SIZE, shuffle=True)


    # Inicialización del modelo student con los mismos parámetros que el teacher
    model = MNISTStudent(
        image_size=MODEL_IMAGE_SIZE,
        in_channels=MODEL_IN_CHANNELS,
        time_embedding_dim=256,  # mismo valor que en teacher
        timesteps=TIMESTEPS,
        base_dim=MODEL_BASE_DIM,         # por ejemplo, 64
        dim_mults=MODEL_DIM_MULTS        # por ejemplo, [2, 4]
    ).to(device)

    # Configuración del EMA, optimizador y scheduler (método similar al teacher)
    adjust = 1 * BATCH_SIZE * MODEL_EMA_STEPS / EPOCHS_STUDENT
    alpha = 1.0 - MODEL_EMA_DECAY
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, lr, total_steps=epochs * len(dataloader), pct_start=0.25, anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')

    # Asegurarse de que exista el directorio de salida para student
    os.makedirs(SAVE_STUDENT_DATA_DIR, exist_ok=True)
    global_steps = 0
    for epoch in range(epochs):
        model.train()
        for step, (images, _) in enumerate(dataloader):
            images = images.to(device)
            noise = torch.randn_like(images).to(device)
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
                print(f"Epoch[{epoch+1}/{epochs}], Step[{step}/{len(dataloader)}], Loss: {loss.item():.5f}, LR: {scheduler.get_last_lr()[0]:.5f}")
        # Al finalizar cada época, se generan imágenes sintéticas con el modelo suavizado
        ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}
        model_ema.eval()
        samples = model_ema.module.sampling(N_SAMPLES_TRAIN, clipped_reverse_diffusion=True, device=device)
        save_image(samples, os.path.join(SAVE_STUDENT_DATA_DIR, f"epoch_{epoch+1}.png"),
                   nrow=int(math.sqrt(N_SAMPLES_TRAIN)), normalize=True)
    # Guardar el checkpoint final del modelo student (similar a como lo hace el teacher)
    torch.save(ckpt, MODEL_PATH_STUDENT)
    return model

if __name__ == "__main__":
    train_student()
