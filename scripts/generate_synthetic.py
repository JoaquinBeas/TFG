import torch
import os
from models.mnist_teacher import MNISTDiffusion
from torchvision.utils import save_image

# Configuración
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 1000
OUTPUT_DIR = "data/synthetic"
MODEL_PATH = "data/train_epochs/epoch_100.pt"

# Generar imágenes sintéticas
def generate_synthetic_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = MNISTDiffusion(image_size=28, in_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)["model"])
    model.eval()

    with torch.no_grad():
        samples = model.sampling(N_SAMPLES, device=DEVICE)

    for i in range(N_SAMPLES):
        save_image(samples[i], f"{OUTPUT_DIR}/sample_{i}.png", normalize=True)

    print(f"Generadas {N_SAMPLES} imágenes sintéticas en {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_synthetic_data()
