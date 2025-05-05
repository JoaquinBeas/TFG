import os
import torch
from torchvision.utils import save_image
from src.utils.config import DEVICE, MODEL_IMAGE_SIZE, MODEL_IN_CHANNELS, TIMESTEPS
from src.diffusion_models.diffusion_unet import DiffusionUnet

# Directorio base de salida
BASE_OUT_DIR = os.path.join("src", "data", "tests", "noise_variation")
os.makedirs(BASE_OUT_DIR, exist_ok=True)

# Configuraciones de prueba
x_t_scales = [0.5, 1.0, 1.5]
noise_scales = [0.5, 1.0, 1.5]
samples_per_combo = 5

# Inicializar y cargar modelo
model = DiffusionUnet(
    image_size=MODEL_IMAGE_SIZE,
    in_channels=MODEL_IN_CHANNELS,
    timesteps=TIMESTEPS
).to(DEVICE)

ckpt_path = os.path.join("src", "data", "train_diffusion_model","diffusion_unet", "last_model.pt")
state = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state.get("model", state))
model.eval()

@torch.no_grad()
def generate_variation():
    for x_t in x_t_scales:
        for noise in noise_scales:
            combo_name = f"x_t{x_t}__noise{noise}"
            out_dir = os.path.join(BASE_OUT_DIR, combo_name)
            os.makedirs(out_dir, exist_ok=True)
            print(f"🔧 Generando para {combo_name}...")

            # Generar muestras
            samples = model.sampling(
                n_samples=samples_per_combo,
                device=DEVICE,
                x_t_scale=x_t,
                noise_scale=noise
            )

            # Guardar imágenes
            for idx, img in enumerate(samples):
                filename = os.path.join(out_dir, f"sample_{idx}.png")
                save_image(img, filename, normalize=True)
            print(f"✅ Guardadas {samples_per_combo} imágenes en {out_dir}")

if __name__ == "__main__":
    generate_variation()
