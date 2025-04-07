import sys
import torch
import os
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
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
from src.utils.training_plot import plot_training_curves

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

def train_student(model_type=StudentModelType.MNIST_STUDENT_COPY, epochs=EPOCHS_STUDENT, lr=LEARNING_RATE, device=DEVICE, patience=8, min_delta=1e-6):
    # Dataset y DataLoader
    dataset = SyntheticNoiseDataset(SYNTHETIC_DIR, OUTPUT_LABELED_DIR)
    synthetic_dataset_repeated = ConcatDataset([dataset] *10)

    # Calcular los tamaños para la división 85% / 15%
    total_len = len(synthetic_dataset_repeated)
    train_len = int(0.85 * total_len)
    val_len = total_len - train_len

    # Dividir el dataset usando random_split
    train_dataset, val_dataset = random_split(
        synthetic_dataset_repeated, 
        [train_len, val_len], 
        generator=torch.Generator().manual_seed(42)  # Para reproducibilidad
    )

    # Crear los dataloaders para cada conjunto
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Seleccionar el modelo según el valor del Enum
    if model_type == StudentModelType.MNIST_STUDENT_COPY:
        from src.models.mnist_teacher import MNISTDiffusion as ModelClass
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
    scheduler = OneCycleLR(optimizer, lr, total_steps=epochs * len(train_loader), pct_start=0.25, anneal_strategy='cos')
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
                print(f"Epoch[{epoch+1}/{epochs}], Step[{step}/{len(train_loader)}], Loss: {loss.item():.8f}, LR: {scheduler.get_last_lr()[0]:.8f}")
        # Al finalizar cada época, se generan imágenes sintéticas con el modelo suavizado
        ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}
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
        # Guardar la imagen en la carpeta específica del modelo (sobrescribiendo si ya existe)
        save_image(samples, os.path.join(SAVE_STUDENT_IMAGES_DIR, f"epoch_{epoch+1}.png"),
                   nrow=int(math.sqrt(N_SAMPLES_TRAIN)), normalize=True)
        torch.save(ckpt, os.path.join(SAVE_MODELS_STUDENT_DIR, f"model_student_{epoch+1}.pt"))
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            counter = 0
            # Guardar el mejor modelo
            best_model = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            best_model_ema = {key: value.cpu().clone() for key, value in model_ema.state_dict().items()}
            # Guardar el mejor modelo en un archivo separado
            best_ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict(), "epoch": epoch+1}
            torch.save(best_ckpt, os.path.join(SAVE_MODELS_STUDENT_DIR, "best_model_student.pt"))
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
    torch.save(best_ckpt, os.path.join(LAST_STUDENT_CKPT))
    # Generar imágenes con el mejor modelo
    model_ema.eval()
    final_samples = model_ema.module.sampling(N_SAMPLES_TRAIN, clipped_reverse_diffusion=True, device=device)
    save_image(final_samples, os.path.join(SAVE_STUDENT_IMAGES_DIR, "final_samples.png"),
       nrow=int(math.sqrt(N_SAMPLES_TRAIN)), normalize=True)

    curve_path = os.path.join(SAVE_STUDENT_IMAGES_DIR, "training_curve.png")
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
