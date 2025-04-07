import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from torchvision.utils import save_image
from collections import Counter
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, DataLoader
from src.config import *
from src.models.mnist_teacher import MNISTDiffusion
from src.utils.data_loader import get_mnist_prototypes

TEMP_DIR = os.path.join("experiments", "temp_clipped_vs_unclipped")
CLIPPED_DIR = os.path.join(TEMP_DIR, "clipped")
UNCLIPPED_DIR = os.path.join(TEMP_DIR, "unclipped")
N_SAMPLES = 1000


# Basicamente, al aplicar ruido usamos ruido puro obtendremos x(t), y si queremos que la imagen este limpia usaremos la reversa x(0)=p(x(t−1)​∣x(t)).
# La version sin clipping que tenemos usa la prediccion directa del ruido para calcular el valor medio del ruido basandose en x(t) y la prediccion de ruido unicamente,
# es simple y eficaz con modelos que predicen muy bien el ruido, el problema es que puede generar valores fuera del rango valido. 
# La version con clipping por otro lado usa la prediccion de x(0) clipeada entre -1 y 1, de esta manera es mas estable y garantiza que las imagenes generadas esten dentro,
# del rango visual correcto.
# Segun estos datos, en nuestro caso, al predecir sobre MNIST, la diferencia entre usar uno u otro es de poca imporancia dado que mayormente afecta a la calidad de la imagen
# generada en cuanto a la claridad de la imagen, nitidez y aparicion de ruido en puntos de la imagen, y en nuestro caso la diferencia en accuracy entre uno y otro a la hora 
# de predecir una label sera practicamente nula, pero en caso de aplicar sampling a una imagen compleja las diferencias serian mucho mas notorias.

class SyntheticImageDataset(Dataset):
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

def label_batch(images, model, prototypes):
    noise = torch.randn_like(images).to(DEVICE)
    with torch.no_grad():
        pred_noise = model(images, noise)
        reconstructed = images - pred_noise

    B = reconstructed.size(0)
    flat = reconstructed.view(B, -1)
    proto_flat = prototypes.view(10, -1)
    dists = torch.cdist(flat, proto_flat)
    return torch.argmin(dists, dim=1).tolist()

def run_experiment():
    os.makedirs(CLIPPED_DIR, exist_ok=True)
    os.makedirs(UNCLIPPED_DIR, exist_ok=True)

    # 1. Cargar modelo
    model = MNISTDiffusion(
        image_size=MODEL_IMAGE_SIZE,
        in_channels=MODEL_IN_CHANNELS,
        base_dim=MODEL_BASE_DIM,
        dim_mults=MODEL_DIM_MULTS
    ).to(DEVICE)
    model.load_state_dict(torch.load(LAST_TEACHER_CKPT, map_location=DEVICE)["model"])
    model.eval()

    # 2. Sampling
    print("🎯 Generando imágenes...")
    with torch.no_grad():
        samples_clipped = model.sampling(N_SAMPLES, clipped_reverse_diffusion=True, device=DEVICE)
        samples_unclipped = model.sampling(N_SAMPLES, clipped_reverse_diffusion=False, device=DEVICE)

    for i in range(N_SAMPLES):
        save_image(samples_clipped[i], os.path.join(CLIPPED_DIR, f"sample_{i}.png"), normalize=True)
        save_image(samples_unclipped[i], os.path.join(UNCLIPPED_DIR, f"sample_{i}.png"), normalize=True)

    # 3. Prototipos
    prototypes = get_mnist_prototypes().to(DEVICE)

    # 4. Dataset y Dataloader
    transform = Compose([Resize((28, 28)), ToTensor()])
    clipped_loader = DataLoader(SyntheticImageDataset(CLIPPED_DIR, transform), batch_size=64)
    unclipped_loader = DataLoader(SyntheticImageDataset(UNCLIPPED_DIR, transform), batch_size=64)

    # 5. Etiquetar → comparar
    print("🔖 Etiquetando y evaluando...")

    def eval_accuracy(loader):
        correct = 0
        total = 0
        for batch_imgs, _ in loader:
            batch_imgs = batch_imgs.to(DEVICE)
            labels_assigned = label_batch(batch_imgs, model, prototypes)
            labels_predicted = label_batch(batch_imgs, model, prototypes)
            correct += sum([1 for a, b in zip(labels_assigned, labels_predicted) if a == b])
            total += len(labels_assigned)
        return correct / total if total > 0 else 0

    acc_clipped = eval_accuracy(clipped_loader)
    acc_unclipped = eval_accuracy(unclipped_loader)

    print(f"\n📊 Accuracy Clipped:   {acc_clipped*100:.2f}%")
    print(f"📊 Accuracy Unclipped: {acc_unclipped*100:.2f}%")

    # Cleanup
    print("🧹 Limpiando imágenes generadas...")
    import shutil
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

# --------------------------
if __name__ == "__main__":
    run_experiment()
