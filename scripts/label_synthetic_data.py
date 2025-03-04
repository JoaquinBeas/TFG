import sys
import torch
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))#TODO:
from torchvision.transforms import ToTensor
from PIL import Image
from models.mnist_teacher import MNISTDiffusion

# Configuración
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYNTHETIC_DIR = "src/data/synthetic"
OUTPUT_DIR = "src/data/labeled_synthetic"
MODEL_PATH = "src/data/train/teacher_epochs/epoch_40.pt" #TODO:

def load_teacher_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model_config = checkpoint.get("model_config", {"image_size": 28, "in_channels": 1, "base_dim": 64, "dim_mults": [2, 4]})
    model = MNISTDiffusion(**model_config).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def label_synthetic_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_teacher_model()

    for filename in os.listdir(SYNTHETIC_DIR):
        if filename.endswith(".png"):
            image_path = os.path.join(SYNTHETIC_DIR, filename)
            image = Image.open(image_path).convert("L")
            image = ToTensor()(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                label = model(image, noise=torch.randn_like(image).to(DEVICE))

            label = torch.argmax(label).item()

            with open(os.path.join(OUTPUT_DIR, f"{filename}.txt"), "w") as f:
                f.write(str(label))

    print(f"Etiquetadas imágenes sintéticas en {OUTPUT_DIR}")

if __name__ == "__main__":
    label_synthetic_data()
