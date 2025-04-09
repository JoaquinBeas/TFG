import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import os
import shutil
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image
from collections import Counter
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, DataLoader

from src.config import *
from src.utils.data_loader import get_mnist_prototypes
from src.models.mnist_teacher import MNISTDiffusion

# Existe una desviacion basada en dos razones:
# 1 para que la varianza tienda a 0, las muestras han de tender a infinito
# 2 un modelo no bien entrenado (degradado) asigna labels equivocas tendiendo a unas sobre otras, lo que impide la correcta representacion de las clases.


TEMP_DIR = os.path.join("experiments", "temp_sampling")
TEMP_LABELS = OUTPUT_LABELED_DIR
BATCH_SIZE = 1000
TOTAL = 50000

class SyntheticImageDataset(Dataset):
    def __init__(self, synthetic_dir, transform=None):
        self.synthetic_dir = synthetic_dir
        self.files = sorted(
            [f for f in os.listdir(synthetic_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        image_path = os.path.join(self.synthetic_dir, filename)
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, filename

def load_teacher_model():
    checkpoint = torch.load(os.path.join(SAVE_MODELS_TEACHER_DIR, "model_teacher_last.pt"), map_location=DEVICE)
    model = MNISTDiffusion(
        image_size=MODEL_IMAGE_SIZE,
        in_channels=MODEL_IN_CHANNELS,
        base_dim=MODEL_BASE_DIM,
        dim_mults=MODEL_DIM_MULTS
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

@torch.no_grad()
def generate_images(model):
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"\n📷 Generando {TOTAL} imágenes en lotes de {BATCH_SIZE}...")
    for offset in tqdm(range(0, TOTAL, BATCH_SIZE), desc="Sampling"):
        samples = model.sampling(BATCH_SIZE, clipped_reverse_diffusion=True, device=DEVICE)
        for i in range(BATCH_SIZE):
            path = os.path.join(TEMP_DIR, f"sample_{offset + i}.png")
            save_image(samples[i], path, normalize=True)

def reconstruct_and_label(batch, model, prototypes):
    noise = torch.randn_like(batch).to(DEVICE)
    with torch.no_grad():
        pred_noise = model(batch, noise)
        reconstructed = batch - pred_noise

    B = reconstructed.size(0)
    reconstructed_flat = reconstructed.view(B, -1)
    prototypes_flat = prototypes.view(10, -1)
    distances = torch.cdist(reconstructed_flat, prototypes_flat, p=2)
    labels = torch.argmin(distances, dim=1).tolist()
    return labels

def label_images(model):
    print("🔖 Etiquetando imágenes...")
    transform = Compose([Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)), ToTensor()])
    dataset = SyntheticImageDataset(TEMP_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    prototypes = get_mnist_prototypes().to(DEVICE)

    os.makedirs(OUTPUT_LABELED_DIR, exist_ok=True)

    for batch_imgs, filenames in tqdm(dataloader, desc="Etiquetando"):
        batch_imgs = batch_imgs.to(DEVICE)
        labels = reconstruct_and_label(batch_imgs, model, prototypes)
        for filename, label in zip(filenames, labels):
            file_name_no_ext = os.path.splitext(filename)[0]
            out_path = os.path.join(OUTPUT_LABELED_DIR, f"{file_name_no_ext}.txt")
            with open(out_path, "w") as f:
                f.write(str(label))

def analyze_labels():
    label_counts = Counter()
    print("\n📊 Analizando distribución de etiquetas...")
    for filename in os.listdir(TEMP_LABELS):
        if filename.endswith(".txt"):
            path = os.path.join(TEMP_LABELS, filename)
            with open(path, "r") as f:
                label = f.read().strip()
                if label.isdigit():
                    label_counts[int(label)] += 1

    total = sum(label_counts.values())
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        print(f"Clase {label}: {count} muestras ({100 * count / total:.2f}%)")

    plt.bar(label_counts.keys(), label_counts.values())
    plt.title("Distribución de clases sintéticas")
    plt.xlabel("Clase")
    plt.ylabel("Frecuencia")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def cleanup():
    print("\n🧹 Eliminando imágenes generadas...")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    model = load_teacher_model()
    generate_images(model)
    label_images(model)
    analyze_labels()
    # cleanup()
