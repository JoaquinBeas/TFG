import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import subprocess
import torch
from torchvision.utils import save_image
from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from src.config import *
from src.models.mnist_teacher import MNISTDiffusion
from src.utils.data_loader import get_mnist_prototypes

TEMP_DIR = os.path.join("experiments", "temp_degraded_eval")
MODELS_NUM = 5
os.makedirs(TEMP_DIR, exist_ok=True)

# Podemos observar que la precision mejora conforme mejora el modelo

class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".png")], key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.folder, filename)
        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, filename

@torch.no_grad()
def label_batch(images, model, prototypes):
    noise = torch.randn_like(images).to(DEVICE)
    pred_noise = model(images, noise)
    reconstructed = images - pred_noise

    flat = reconstructed.view(images.size(0), -1)
    proto_flat = prototypes.view(10, -1)
    dists = torch.cdist(flat, proto_flat)
    return torch.argmin(dists, dim=1).tolist()

def load_model(ckpt_path):
    model = MNISTDiffusion(
        image_size=MODEL_IMAGE_SIZE,
        in_channels=MODEL_IN_CHANNELS,
        base_dim=MODEL_BASE_DIM,
        dim_mults=MODEL_DIM_MULTS
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model"])
    model.eval()
    return model

def get_equidistant_checkpoints(n=MODELS_NUM):
    all_files = [f for f in os.listdir(SAVE_MODELS_TEACHER_DIR) if f.startswith("model_teacher_") and f.endswith(".pt") and not "last" in f]
    epochs = sorted([int(f.split("_")[-1].split(".")[0]) for f in all_files if f.split("_")[-1].split(".")[0].isdigit()])
    if len(epochs) < n - 1:
        selected_epochs = epochs
    else:
        step = len(epochs) // (n - 1)
        selected_epochs = [epochs[i * step] for i in range(n - 1)]

    # Construir paths para modelos numerados
    paths = [os.path.join(SAVE_MODELS_TEACHER_DIR, f"model_teacher_{e}.pt") for e in selected_epochs]

    # Añadir explícitamente model_teacher_last.pt al final
    paths.append(os.path.join(SAVE_MODELS_TEACHER_DIR, "model_teacher_last.pt"))
    return paths


def evaluate_accuracy_against_final(model_to_test, images, model_final, prototypes):
    labels_test = label_batch(images, model_to_test, prototypes)
    labels_final = label_batch(images, model_final, prototypes)
    correct = sum([1 for a, b in zip(labels_test, labels_final) if a == b])
    return correct / len(labels_test)


def generate_samples(model, out_dir, n_samples=500):
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        samples = model.sampling(n_samples, clipped_reverse_diffusion=True, device=DEVICE)
        for i in range(n_samples):
            save_image(samples[i], os.path.join(out_dir, f"sample_{i}.png"), normalize=True)

if __name__ == "__main__":
    # Paso 1: Verificar si existe el modelo entrenado
    if not os.path.exists(LAST_TEACHER_CKPT):
        print("📦 No se encontró model_teacher_last.pt. Entrenando...")
        subprocess.run(["python", "scripts/train_teacher.py"], check=True)

    # Paso 2: Seleccionar 5 checkpoints
    checkpoints = get_equidistant_checkpoints(n=5)

    print("\n🔍 Checkpoints seleccionados:")
    for ckpt in checkpoints:
        print(" -", os.path.basename(ckpt))

    results = []

    # Cargar modelo final una sola vez
    final_model = load_model(LAST_TEACHER_CKPT)
    prototypes = get_mnist_prototypes().to(DEVICE)

    for ckpt_path in checkpoints:
        epoch = "last" if "last" in ckpt_path else os.path.basename(ckpt_path).split("_")[-1].split(".")[0]
        print(f"\n🚀 Evaluando modelo de la época {epoch}...")

        model = load_model(ckpt_path)
        sample_dir = os.path.join(TEMP_DIR, f"samples_epoch_{epoch}")
        generate_samples(model, sample_dir, n_samples=500)

        # Preparar dataloader de imágenes
        dataset = ImageDataset(sample_dir, transform=Compose([Resize((28, 28)), ToTensor()]))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        # Acumular imágenes en memoria
        all_images = []
        for batch_imgs, _ in dataloader:
            all_images.append(batch_imgs)
        all_images = torch.cat(all_images).to(DEVICE)

        acc = evaluate_accuracy_against_final(model, all_images, final_model, prototypes)
        results.append((epoch, acc))

    print("\n📊 Resultados de accuracy comparado con el modelo final:")
    print("Epoch\tAccuracy vs Final")
    print("-" * 30)
    for epoch, acc in results:
        print(f"{epoch}\t{acc * 100:.2f}%")

    # Limpieza
    import shutil
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
