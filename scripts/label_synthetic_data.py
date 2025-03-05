import sys
import torch
import os
import setup_paths
from torchvision.transforms import ToTensor
from PIL import Image
from models.mnist_teacher import MNISTDiffusion
from config import *

def load_teacher_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model_config = checkpoint.get("model_config", {"image_size": MODEL_IMAGE_SIZE, "in_channels": MODEL_IN_CHANNELS, "base_dim": MODEL_BASE_DIM, "dim_mults": MODEL_DIM_MULTS})
    model = MNISTDiffusion(**model_config).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def label_synthetic_data():
    os.makedirs(OUTPUT_LABELED_DIR, exist_ok=True)
    model = load_teacher_model()
    for filename in os.listdir(SYNTHETIC_DIR):
        if filename.endswith(".png"):
            image_path = os.path.join(SYNTHETIC_DIR, filename)
            image = Image.open(image_path).convert("L")
            image = ToTensor()(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                label = model(image, noise=torch.randn_like(image).to(DEVICE))
            label = torch.argmax(label).item()
            with open(os.path.join(OUTPUT_LABELED_DIR, f"{filename}.txt"), "w") as f:
                f.write(str(label))
    print(f"Etiquetadas imágenes sintéticas en {OUTPUT_LABELED_DIR}")

if __name__ == "__main__":
    label_synthetic_data()
