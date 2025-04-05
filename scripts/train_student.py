import sys
from sklearn.model_selection import KFold
import torch
import os
from torch.utils.data import DataLoader, Dataset,random_split
import torch.nn as nn
from torch.optim import AdamW
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from torchvision.utils import save_image
from torch.optim.lr_scheduler import OneCycleLR
from enum import Enum
import math
from src.utils.training_plot import plot_training_curves
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
    # ==== División fija 85% train / 15% val ====
    dataset = SyntheticNoiseDataset(SYNTHETIC_DIR, OUTPUT_LABELED_DIR)
    total_size = len(dataset)
    val_size = int(0.15 * total_size)
    train_size = total_size - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # === Selección de modelo ===
    if model_type == StudentModelType.MNIST_STUDENT_COPY:
        from src.models.mnist_student_copy import MNISTStudent as ModelClass
    elif model_type == StudentModelType.MNIST_STUDENT_RESNET:
        from src.models.mnist_student_resnet import MNISTStudentResNet as ModelClass
    elif model_type == StudentModelType.MNIST_STUDENT_GUIDED:
        from src.models.mnist_student_guided import MNISTStudentGuided as ModelClass
    else:
        raise ValueError(f"Modelo desconocido: {model_type}")

    model = ModelClass(
        image_size=MODEL_IMAGE_SIZE,
        in_channels=MODEL_IN_CHANNELS,
        time_embedding_dim=256,
        timesteps=TIMESTEPS
    ).to(device)

    adjust = 1 * BATCH_SIZE * MODEL_EMA_STEPS / EPOCHS_STUDENT
    alpha = 1.0 - MODEL_EMA_DECAY
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, lr, total_steps=epochs * len(train_loader), pct_start=0.25, anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')

    os.makedirs(SAVE_STUDENT_IMAGES_DIR, exist_ok=True)
    os.makedirs(SAVE_MODELS_STUDENT_DIR, exist_ok=True)

    if CKTP:
        cktp = torch.load(CKTP)
        model_ema.load_state_dict(cktp["model_ema"])
        model.load_state_dict(cktp["model"])

    global_steps = 0
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 7

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for step, (images, _) in enumerate(train_loader):
            images = images.to(device)
            noise = torch.randn_like(images).to(device)
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
                print(f"Epoch[{epoch+1}/{epochs}], Step[{step}/{len(train_loader)}], Loss: {loss.item():.5f}, LR: {scheduler.get_last_lr()[0]:.5f}")
     
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === VALIDACIÓN ===
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                noise = torch.randn_like(images).to(device)
                pred_noise = model(images, noise)
                loss = loss_fn(pred_noise, noise)
                val_loss_total += loss.item()
                val_steps += 1
        avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0.0
        val_losses.append(avg_val_loss)
        print(f"Epoch[{epoch+1}] - Validation Loss: {avg_val_loss:.5f}")

        # EARLY STOPPING
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print("Detenido por early stopping.")
                break

    # Guardar mejor modelo
    if best_val_loss < float('inf'):
        model.load_state_dict(best_model_state)
        ckpt = {"model": model.state_dict()}
        best_path = os.path.join(SAVE_MODELS_STUDENT_DIR, "model_student_best.pt")
        torch.save(ckpt, best_path)
        print(f"\nMejor modelo del Student guardado en: {best_path}")
    curve_path = os.path.join(SAVE_TEACHER_IMAGES_DIR, "training_curve.png")    
    plot_training_curves(train_losses, val_losses, curve_path)
    print(f"Gráfica de entrenamiento guardada en: {curve_path}")
    return model

if __name__ == "__main__":
    # Se puede pasar el nombre del modelo a través de un argumento de línea de comandos
    try:
        model_arg = sys.argv[1].upper()  # Convertir a mayúsculas para facilitar la comparación
        model_type = StudentModelType[model_arg]
    except Exception:
        model_type = StudentModelType.MNIST_STUDENT_COPY
    train_student(model_type=model_type)
