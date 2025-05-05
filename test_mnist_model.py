#!/usr/bin/env python3
import torch
from src.train_mnist_model import MnistTrainer
from src.utils.mnist_models_enum import MNISTModelType

def main():
    # Indicamos que se utilizará el dataset MNIST original
    use_synthetic_dataset = False

    # Instanciamos el entrenador para el modelo MNIST complejo (MNISTNet1)
    trainer = MnistTrainer(
        model_type=MNISTModelType.RESNET_PREACT,  # Selecciona el modelo complejo
        num_epochs=20,                          # Número de épocas (ajusta según convenga)
        learning_rate=0.002,                    # Tasa de aprendizaje
        batch_size=64,                          # Tamaño de lote
        early_stopping_patience=10,             # Patience para early stopping (ajusta si lo deseas)
        use_synthetic_dataset=use_synthetic_dataset
    )

    # Se entrena el modelo y se evalúa sobre el conjunto de test internamente
    avg_loss, accuracy = trainer.model()

    print("\n=== Resultados finales en el conjunto de test ===")
    print(f"Pérdida Promedio: {avg_loss:.4f}")
    print(f"Precisión: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
