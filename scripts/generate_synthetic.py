import sys
import torch
import os
from torchvision.utils import save_image

from src.models.mnist_teacher import MNISTDiffusion
from src.config import *

# Descomentar para ejecutar desde aqui
# from models.mnist_teacher import MNISTDiffusion  
# from config import *

# Generar imágenes sintéticas
def generate_synthetic_data():
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    model = MNISTDiffusion(image_size=MODEL_IMAGE_SIZE, in_channels=MODEL_IN_CHANNELS, base_dim=MODEL_BASE_DIM, dim_mults=MODEL_DIM_MULTS).to(DEVICE)
    model.load_state_dict(torch.load(LAST_TEACHER_CKPT, map_location=DEVICE)["model"])
    model.eval()
    with torch.no_grad():
        samples = model.sampling(N_SAMPLES_GENERATE, device=DEVICE)
    for i in range(N_SAMPLES_GENERATE):
        save_image(samples[i], f"{SYNTHETIC_DIR}/sample_{i}.png", normalize=True)
    print(f"Generadas {N_SAMPLES_GENERATE} imágenes sintéticas en {SYNTHETIC_DIR}")

if __name__ == "__main__":
    generate_synthetic_data()
