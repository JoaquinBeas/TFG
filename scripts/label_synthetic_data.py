import torch
import os
import setup_paths
from torchvision.transforms import ToTensor
from PIL import Image
from models.mnist_teacher import MNISTDiffusion
from config import *
from data.data_loader import get_mnist_prototypes  # Función que carga ejemplos de los números 0-9

def load_teacher_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = MNISTDiffusion(image_size=MODEL_IMAGE_SIZE, in_channels=MODEL_IN_CHANNELS, base_dim=MODEL_BASE_DIM, dim_mults=MODEL_DIM_MULTS).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def reconstruct_and_label(image, model):
    """ 
    Reconstruye la imagen eliminando el ruido predicho y asigna la clase más similar. 
    """
    noise = torch.randn_like(image).to(DEVICE)
    with torch.no_grad():
        pred_noise = model(image, noise)  # Predice el ruido
        reconstructed = image - pred_noise  # Se resta el ruido para obtener la imagen reconstruida

    # Cargar prototipos de MNIST (imágenes promedio de cada dígito 0-9)
    prototypes = get_mnist_prototypes().to(DEVICE)  # (10, 1, 28, 28)

    # Medir la distancia entre la imagen reconstruida y los prototipos
    distances = torch.linalg.norm((prototypes - reconstructed).view(10, -1), dim=1)
    label = torch.argmin(distances).item()  # La imagen más cercana en distancia define la clase
    return label

def label_synthetic_data():
    os.makedirs(OUTPUT_LABELED_DIR, exist_ok=True)
    model = load_teacher_model()
    files = sorted(
        [f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".png")], 
        key=extract_number
    )

    # Procesar los archivos en orden numérico
    for filename in files:
        if filename.endswith(".png"):
            image_path = os.path.join(SYNTHETIC_DIR, filename)
            image = Image.open(image_path).convert("L")
            image = ToTensor()(image).unsqueeze(0).to(DEVICE)

            label = reconstruct_and_label(image, model)
            file_name_no_ext = os.path.splitext(filename)[0]
            with open(os.path.join(OUTPUT_LABELED_DIR, f"{file_name_no_ext}.txt"), "w") as f:
                f.write(str(label))

    print(f"Etiquetadas imágenes sintéticas en {OUTPUT_LABELED_DIR}")
    
def extract_number(filename):
    return int(filename.split("_")[-1].split(".")[0])

if __name__ == "__main__":
    label_synthetic_data()
