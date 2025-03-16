import os
import torch
import setup_paths
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
from models.mnist_teacher import MNISTDiffusion
from config import *
from data.data_loader import get_mnist_prototypes
from torch.utils.data import Dataset, DataLoader

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
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = MNISTDiffusion(
        image_size=MODEL_IMAGE_SIZE, 
        in_channels=MODEL_IN_CHANNELS, 
        base_dim=MODEL_BASE_DIM, 
        dim_mults=MODEL_DIM_MULTS
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def reconstruct_and_label(batch, model, prototypes):
    """
    Procesa un batch de imágenes.
    batch: tensor de imágenes [B, 1, 28, 28]
    prototypes: tensor de prototipos [10, 1, 28, 28]
    Devuelve: lista de etiquetas para cada imagen del batch.
    """
    noise = torch.randn_like(batch).to(DEVICE)
    with torch.no_grad():
        pred_noise = model(batch, noise)  # Predice el ruido
        reconstructed = batch - pred_noise  # Reconstruye la imagen

    # Calcula distancias entre cada imagen reconstruida y cada prototipo
    B = reconstructed.size(0)
    # Aplanamos cada imagen para la comparación (cada imagen: [1,28,28] -> [28*28])
    reconstructed_flat = reconstructed.view(B, -1)
    prototypes_flat = prototypes.view(10, -1)  # [10, 28*28]
    
    # Calculamos las distancias usando broadcasting: [B, 10]
    distances = torch.cdist(reconstructed_flat, prototypes_flat, p=2)
    labels = torch.argmin(distances, dim=1).tolist()
    return labels

def label_synthetic_data():
    # Definir transformaciones (para asegurar tamaño 28x28 y convertir a tensor)
    transform = Compose([
        Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)),
        ToTensor()
    ])

    # Crear dataset y dataloader (ajusta num_workers según tu sistema)
    dataset = SyntheticImageDataset(SYNTHETIC_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model = load_teacher_model()
    prototypes = get_mnist_prototypes().to(DEVICE)  # [10, 1, 28, 28]

    os.makedirs(OUTPUT_LABELED_DIR, exist_ok=True)

    for batch_imgs, filenames in dataloader:
        batch_imgs = batch_imgs.to(DEVICE)
        labels = reconstruct_and_label(batch_imgs, model, prototypes)
        # Guarda cada etiqueta en su archivo correspondiente
        for filename, label in zip(filenames, labels):
            file_name_no_ext = os.path.splitext(filename)[0]
            out_path = os.path.join(OUTPUT_LABELED_DIR, f"{file_name_no_ext}.txt")
            with open(out_path, "w") as f:
                f.write(str(label))
    print(f"Etiquetadas imágenes sintéticas en {OUTPUT_LABELED_DIR}")

if __name__ == "__main__":
    label_synthetic_data()
