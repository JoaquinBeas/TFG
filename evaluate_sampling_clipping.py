# evaluate_sampling_clipping.py

import os
import torch
import shutil
from collections import Counter
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from src.utils.config import DEVICE, TRAIN_MNIST_MODEL_DIR, TRAIN_DIFFUSION_MODEL_DIR, DATA_DIR
from src.train_mnist_model import MnistTrainer, MNISTModelType
from src.train_diffussion_model import DiffussionTrainer, DiffusionModelType

# Directorio base 
TEST_BASE = os.path.join(DATA_DIR, "tests", "evaluate_sampling") 
CLIPPED_DIR = os.path.join(TEST_BASE, "clipped")
UNCLIPPED_DIR = os.path.join(TEST_BASE, "unclipped")
N_IMAGES = 20
BATCH_SIZE = 128

# Transforms para pasar PIL→tensor normalizado igual que en training
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def prepare_dirs():
    # Borra sólo si queremos reiniciar el test, 
    # aquí aseguramos que existan las carpetas limpias
    for d in (CLIPPED_DIR, UNCLIPPED_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

def generate_samples_for_diffusion(diff_type: DiffusionModelType):
    """
    Carga o entrena el modelo de difusión y genera N_IMAGES muestras
    clipped y unclipped en las carpetas correspondientes.
    """
    model_dir = os.path.join(TRAIN_DIFFUSION_MODEL_DIR, diff_type.value)
    ckpt = os.path.join(model_dir, "last_model.pt")
    # Entrena si no existe
    if not os.path.isfile(ckpt):
        trainer = DiffussionTrainer(model_type=diff_type, num_epochs=10, batch_size=128)
        trainer.train_model()
    # Carga el modelo
    trainer = DiffussionTrainer(model_type=diff_type, num_epochs=20)  # epochs=0 sólo para instanciar
    state = torch.load(ckpt, map_location=DEVICE)
    trainer.model.load_state_dict(state, strict=False)
    trainer.model.to(DEVICE).eval()

    # Muestreo
    clipped = trainer.model.sampling(N_IMAGES, clipped_reverse_diffusion=True, device=DEVICE)
    unclipped = trainer.model.sampling(N_IMAGES, clipped_reverse_diffusion=False, device=DEVICE)

    # Guarda PNGs
    for i, img in enumerate(clipped):
        transforms.ToPILImage()(img.cpu()).save(os.path.join(CLIPPED_DIR, f"{diff_type.value}_{i}.png"))
    for i, img in enumerate(unclipped):
        transforms.ToPILImage()(img.cpu()).save(os.path.join(UNCLIPPED_DIR, f"{diff_type.value}_{i}.png"))

def load_or_train_mnist(model_type: MNISTModelType):
    """
    Carga el clasificador MNIST o lo entrena 20 epochs si no hay checkpoint.
    """
    ckpt_dir = os.path.join(TRAIN_MNIST_MODEL_DIR, model_type.value)
    ckpt = os.path.join(ckpt_dir, "last_model.pt")
    if os.path.isfile(ckpt):
        # Reutilizamos la función de utils en main.py
        from main import load_mnist_model
        return load_mnist_model(model_type)
    else:
        trainer = MnistTrainer(model_type=model_type, num_epochs=20)
        trainer.train_model()
        return trainer.get_model()

def eval_confidence(loader, mnist_model):
    """
    Para un DataLoader de imágenes sin etiquetas, devuelve
    la confianza media y la distribución de predicciones.
    """
    mnist_model.to(DEVICE).eval()
    all_confs, all_preds = [], []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            logits = mnist_model(imgs)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_confs.extend(confs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    return sum(all_confs)/len(all_confs), Counter(all_preds)

def run():
    prepare_dirs()

    for diff_type in DiffusionModelType:
        print(f"\n=== Sampling con diffusion model {diff_type.value} ===")
        generate_samples_for_diffusion(diff_type)

        # --- CORRECCIÓN: Clase ImgFolder usa self.folder ---
        class ImgFolder(Dataset):
            def __init__(self, folder):
                self.folder = folder
                self.files = sorted(os.listdir(folder))[:N_IMAGES]
            def __len__(self):
                return len(self.files)
            def __getitem__(self, i):
                path = os.path.join(self.folder, self.files[i])
                img = Image.open(path).convert("L")
                return transform(img), self.files[i]

        # Ahora sí podemos construir los loaders
        clipped_loader = DataLoader(ImgFolder(CLIPPED_DIR), batch_size=BATCH_SIZE)
        unclipped_loader = DataLoader(ImgFolder(UNCLIPPED_DIR), batch_size=BATCH_SIZE)

        mn_type = MNISTModelType.RESNET_PREACT
        print(f"\n--> Evaluando con MNIST model {mn_type.value}")
        mn_model = load_or_train_mnist(mn_type)
        avg_c, dist_c = eval_confidence(clipped_loader, mn_model)
        avg_u, dist_u = eval_confidence(unclipped_loader, mn_model)

        print(f"   • Clipped:   avg_conf={avg_c:.3f}, dist={dict(dist_c)}")
        print(f"   • Unclipped: avg_conf={avg_u:.3f}, dist={dict(dist_u)}")


if __name__ == "__main__":
    run()
