import sys
import torch
import os
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn as nn
from torch.optim import AdamW
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from torchvision.utils import save_image
from torch.optim.lr_scheduler import OneCycleLR
from enum import Enum
import math

from src.utils.exponential_moving_avg import ExponentialMovingAverage
from src.config import *

# Descomentar para ejecutar desde aqui

# from utils.exponential_moving_avg import ExponentialMovingAverage
# from config import *

# Definición del Enum
class StudentModelType(Enum):
    MNIST_STUDENT_COPY = "mnist_student_copy"        # Igual a teacher
    MNIST_STUDENT_RESNET = "mnist_student_resnet"      # ResNet puro sin UNet (muy lento)
    MNIST_STUDENT_GUIDED = "mnist_student_guided"      # Guided diffusion model

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

def train_student(model_type=StudentModelType.MNIST_STUDENT_COPY, epochs=EPOCHS_STUDENT, lr=LEARNING_RATE, device=DEVICE):
    # Dataset y DataLoader
    dataset = SyntheticNoiseDataset(SYNTHETIC_DIR, OUTPUT_LABELED_DIR)
    synthetic_dataset_repeated = ConcatDataset([dataset] * 70)
    dataloader = DataLoader(synthetic_dataset_repeated, batch_size=BATCH_SIZE, shuffle=True)

    # Seleccionar el modelo según el valor del Enum
    if model_type == StudentModelType.MNIST_STUDENT_COPY:
        from src.models.mnist_student_copy import MNISTStudent as ModelClass
    elif model_type == StudentModelType.MNIST_STUDENT_RESNET:
        from src.models.mnist_student_resnet import MNISTStudentResNet as ModelClass
    elif model_type == StudentModelType.MNIST_STUDENT_GUIDED:
        from src.models.mnist_student_guided import MNISTStudentGuided as ModelClass
    else:
        raise ValueError(f"Modelo desconocido: {model_type}")

    # Para el checkpoint de student, podemos usar un nombre basado en el modelo:
    student_checkpoint_name = f"model_{model_type.value}.pt"

    # Crear el modelo. Se envían los parámetros comunes; en el caso de algunos modelos,
    # parámetros como "time_embedding_dim" podrían ser ignorados si no se usan.
    model = ModelClass(
        image_size=MODEL_IMAGE_SIZE,
        in_channels=MODEL_IN_CHANNELS,
        time_embedding_dim=256,
        timesteps=TIMESTEPS
    ).to(device)

    # Configuración del EMA, optimizador y scheduler (similar al teacher)
    adjust = 1 * BATCH_SIZE * MODEL_EMA_STEPS / EPOCHS_STUDENT
    alpha = 1.0 - MODEL_EMA_DECAY
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, lr, total_steps=epochs * len(dataloader), pct_start=0.25, anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')

    # Crear el directorio para guardar imágenes, usando el nombre del modelo
    # Por ejemplo: "mnist_student_copy_epochs", "mnist_student_resnet_epochs", "mnist_student_guided_epochs"
    # Asegurar que exista la carpeta de modelos
    os.makedirs(SAVE_STUDENT_IMAGES_DIR, exist_ok=True)
    os.makedirs(SAVE_MODELS_STUDENT_DIR, exist_ok=True)
    if CKTP:
        cktp=torch.load(CKTP)
        model_ema.load_state_dict(cktp["model_ema"]) #modelo suavizado
        model.load_state_dict(cktp["model"])         #modelo normal 
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
        # Guardar la imagen en la carpeta específica del modelo (sobrescribiendo si ya existe)
        save_image(samples, os.path.join(SAVE_STUDENT_IMAGES_DIR, f"epoch_{epoch+1}.png"),
                   nrow=int(math.sqrt(N_SAMPLES_TRAIN)), normalize=True)
        torch.save(ckpt, os.path.join(SAVE_MODELS_STUDENT_DIR, f"model_student_{epoch+1}.pt"))
    return model

if __name__ == "__main__":
    # Se puede pasar el nombre del modelo a través de un argumento de línea de comandos
    try:
        model_arg = sys.argv[1].upper()  # Convertir a mayúsculas para facilitar la comparación
        model_type = StudentModelType[model_arg]
    except Exception:
        model_type = StudentModelType.MNIST_STUDENT_COPY
    train_student(model_type=model_type)
