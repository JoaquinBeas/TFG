#!/usr/bin/env python
"""
compare_mnist_vs_synthetic.py
-----------------------------
• Carga un modelo ResNet-PreAct ya entrenado con MNIST
  desde 'src/data/train_mnist_model/resnet_preact/last_model.pt'.
• Entrena (30 épocas) un modelo idéntico desde cero
  con el *train set* del dataset sintético.
• Muestra:
      – Accuracy del modelo cargado en test-MNIST
      – Mejor accuracy del modelo sintético en su test
      – Accuracy de ambos modelos sobre test-MNIST
"""

# ---------- CONFIGURACIÓN RÁPIDA ----------
MNIST_MODEL_PATH   = "src/data/train_mnist_model/resnet_preact/last_model.pt"
SYNTHETIC_DATA_DIR = "src/data/generated_datasets/last_model"
NUM_EPOCHS_SYN     = 80
LEARNING_RATE_SYN  = 1e-3
BATCH_SIZE         = 64
# ------------------------------------------

import os
import torch
from src.train_mnist_model import MnistTrainer, MNISTModelType

# ---------- utilidades ----------
def evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


def train_synthetic(trainer) -> float:
    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS_SYN + 1):
        trainer.train_epoch(epoch)
        _, acc = trainer.evaluate_on_loader(
            trainer.test_loader, loader_name="Synthetic-Test"
        )
        best_acc = max(best_acc, acc)
    return best_acc


def main():
    model_type = MNISTModelType.RESNET_PREACT

    # === 1. Cargar modelo MNIST ya entrenado ===
    print("\n== Cargando modelo MNIST pre-entrenado ==")
    trainer_eval = MnistTrainer(          # solo para obtener loaders / device
        model_type=model_type,
        num_epochs=1,
        learning_rate=0.0,
        batch_size=BATCH_SIZE,
    )
    state = torch.load(MNIST_MODEL_PATH, map_location=trainer_eval.device)
    trainer_eval.model.load_state_dict(state)
    trainer_eval.model.eval()

    acc_mnist_test = evaluate_accuracy(
        trainer_eval.model, trainer_eval.test_loader, trainer_eval.device
    )
    print(f"Accuracy modelo MNIST en test-MNIST: {acc_mnist_test:6.2f}%")

    # === 2. Entrenar modelo con dataset sintético ===
    print("\n== Entrenando modelo con dataset sintético ==")
    trainer_syn = MnistTrainer(
        model_type=model_type,
        num_epochs=NUM_EPOCHS_SYN,
        learning_rate=LEARNING_RATE_SYN,
        batch_size=BATCH_SIZE,
        use_synthetic_dataset=True,
        synthetic_data_dir=SYNTHETIC_DATA_DIR,
    )
    best_syn_acc = train_synthetic(trainer_syn)
    print(f"Mejor accuracy en test-SINTÉTICO:     {best_syn_acc:6.2f}%")

    # === 3. Comparación cruzada en test-MNIST ===
    acc_syn_on_mnist = evaluate_accuracy(
        trainer_syn.model, trainer_eval.test_loader, trainer_eval.device
    )

    print("\n== Comparación sobre test set MNIST original ==")
    print(f"Modelo MNIST (cargado)        → {acc_mnist_test:6.2f}%")
    print(f"Modelo entrenado en sintético → {acc_syn_on_mnist:6.2f}%")


if __name__ == "__main__":
    main()
