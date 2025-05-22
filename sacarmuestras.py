#!/usr/bin/env python3
"""
quick_demo_resnet_synth.py

Demostración rápida:
  • Toma 20 muestras aleatorias del dataset sintético (last_model)
  • Entrena ResNet PreAct 10 épocas
  • Muestra predicciones y etiquetas verdaderas
"""

import random
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from src.train_mnist_model import MnistTrainer, MNISTModelType
from src.utils.data_loader import get_synthetic_mnist_dataloaders

# --- CONFIG -----------------------------------------------------------------
SYN_DATA_DIR      = "src/data/generated_datasets/last_model"
BATCH_SIZE_TRAIN  = 64
BATCH_SIZE_DEMO   = 20       # justo las 20 muestras que queremos ver
NUM_EPOCHS        = 10
LR                = 1e-3
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------


def pick_n_random(dataset, n=20, seed=42):
    """Devuelve n índices únicos aleatorios y su DataLoader auxiliar."""
    rng = random.Random(seed)
    idxs = rng.sample(range(len(dataset)), n)
    subset = torch.utils.data.Subset(dataset, idxs)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=n, shuffle=False, num_workers=0
    )
    return idxs, loader


def main():
    # 1) Carga train / test del sintético
    train_loader, test_loader = get_synthetic_mnist_dataloaders(
        batch_size=BATCH_SIZE_TRAIN,
        synthetic_data_dir=SYN_DATA_DIR,
    )

    # 2) Sacamos 20 imágenes aleatorias del *test* (no se usarán para entrenar)
    _, demo_loader = pick_n_random(test_loader.dataset, n=BATCH_SIZE_DEMO)

    # 3) Entrenamos ResNet PreAct 10 épocas
    trainer = MnistTrainer(
        model_type=MNISTModelType.RESNET_PREACT,
        num_epochs=NUM_EPOCHS,
        learning_rate=LR,
        batch_size=BATCH_SIZE_TRAIN,
        use_synthetic_dataset=True,
        synthetic_data_dir=SYN_DATA_DIR
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        trainer.train_epoch(epoch)

    # 4) Obtenemos predicciones sobre las 20 imágenes
    model = trainer.model.to(DEVICE).eval()
    images, labels = next(iter(demo_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)

    # 5) Dibujamos una rejilla 4×5 con labels
    images   = images.cpu().numpy()
    preds    = preds.cpu().numpy()
    labels   = labels.cpu().numpy()

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    axes = axes.ravel()

    for ax, img, pred, true in zip(axes, images, preds, labels):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"P: {pred}  |  T: {true}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if not Path(SYN_DATA_DIR).exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de dataset sintético: {SYN_DATA_DIR}"
        )
    main()
