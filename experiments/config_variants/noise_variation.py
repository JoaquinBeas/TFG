import os
import torch
from torchvision.utils import save_image
from src.config import DEVICE, MODEL_IMAGE_SIZE
from src.models.mnist_teacher import MNISTDiffusion  # O el modelo que estés testeando

# Salida
OUT_DIR = os.path.join("experiments", "sampling_test_output")
os.makedirs(OUT_DIR, exist_ok=True)

# Configs de prueba
x_t_scales = [0.5, 1.0, 1.5]
noise_scales = [0.5, 1.0, 1.5]

# Modelo
model = MNISTDiffusion(
    image_size=MODEL_IMAGE_SIZE,
    in_channels=1,
    base_dim=64,
    dim_mults=[2, 4]
).to(DEVICE)

# Cargar checkpoint
ckpt_path = os.path.join("src", "data", "models_teacher", "model_teacher_last.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model"])
model.eval()

# Generar imágenes por cada combinación
for x_t in x_t_scales:
    for noise in noise_scales:
        print(f"🔧 Generando: x_t={x_t}, noise={noise}")
        samples = model.sampling(n_samples=5, device=DEVICE, x_t_scale=x_t, noise_scale=noise)
        save_dir = os.path.join(OUT_DIR, f"x_{x_t}_noise_{noise}")
        os.makedirs(save_dir, exist_ok=True)
        for i in range(5):
            save_image(samples[i], os.path.join(save_dir, f"sample_{i}.png"), normalize=True)
