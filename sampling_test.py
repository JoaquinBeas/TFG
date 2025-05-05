import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

from src.train_diffussion_model import DiffussionTrainer, DiffusionModelType
from src.train_mnist_model import MnistTrainer, MNISTModelType
from src.utils.config import DEVICE, TRAIN_DIFFUSION_MODEL_DIR, TRAIN_MNIST_MODEL_DIR, MODEL_IMAGE_SIZE
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor

# Número de imágenes a generar y etiquetar
NUM_SAMPLES = 1000
BATCH_SIZE = 100

def get_diffusion_model(model_type: DiffusionModelType):
    checkpoint_dir = os.path.join(TRAIN_DIFFUSION_MODEL_DIR, model_type.value)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "last_model.pt")

    trainer = DiffussionTrainer(model_type=model_type, num_epochs=50)
    if os.path.exists(ckpt_path):
        trainer.model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"Cargado diffusion {model_type.value} de {ckpt_path}")
    else:
        print(f"No existe checkpoint para {model_type.value}, entrenando 50 epochs...")
        trainer.train_model()
    trainer.model.eval()
    return trainer.model

def get_mnist_model():
    model_type = MNISTModelType.RESNET_PREACT
    checkpoint_dir = os.path.join(TRAIN_MNIST_MODEL_DIR, model_type.value)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "last_model.pt")

    trainer = MnistTrainer(model_type=model_type, num_epochs=30)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        trainer.model.load_state_dict(state, strict=False)        
        print(f"Cargado MNIST {model_type.value} de {ckpt_path}")
    else:
        print(f"No existe checkpoint MNIST {model_type.value}, entrenando 30 epochs...")
        trainer.train_model()
    trainer.model.eval()
    return trainer.model

@torch.no_grad()
def sample_and_label(diffusion_model, mnist_model, out_dir: str):
    # Generar imágenes
    os.makedirs(out_dir, exist_ok=True)
    samples = []
    for _ in tqdm(range(NUM_SAMPLES // BATCH_SIZE), desc="Sampling"):
        batch = diffusion_model.sampling(BATCH_SIZE, device=DEVICE)
        samples.append(batch.cpu())
    samples = torch.cat(samples, dim=0)[:NUM_SAMPLES]

    # Guardar temporalmente y etiquetar
    transform = Compose([Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)), ToTensor()])
    labels = []
    for img in samples:
        # Normalizar como en entrenamiento MNIST
        mn = (img - 0.5) / 0.5
        logits = mnist_model(mn.to(DEVICE).unsqueeze(0))
        pred = logits.argmax(dim=1).item()
        labels.append(pred)

    # Calcular distribución
    counts = Counter(labels)
    total = sum(counts.values())
    perc = {cls: counts.get(cls,0) / total * 100 for cls in range(10)}

    # Plot
    plt.figure()
    plt.bar(perc.keys(), perc.values())
    plt.xlabel("Clase")
    plt.ylabel("% muestras")
    plt.title(f"Distribución sintética {diffusion_model.__class__.__name__}")
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"samplesbyclass_{diffusion_model.__class__.__name__}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Guardado gráfico en {save_path}")

def main():
    # Cargar o entrenar MNIST
    mnist_model = get_mnist_model()

    # Para cada tipo de difusión
    for model_type in DiffusionModelType:
        print(f"\n=== Procesando {model_type.value} ===")
        diffusion_model = get_diffusion_model(model_type)
        out_dir = os.path.join("src", "data", "test", "sampling_test")
        sample_and_label(diffusion_model, mnist_model, out_dir)

if __name__ == "__main__":
    main()
