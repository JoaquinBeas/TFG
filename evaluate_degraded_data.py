import matplotlib.pyplot as plt
import pandas as pd
from src.train_mnist_model import MnistTrainer, MNISTModelType

# Ruta al dataset sintético (“last_model” en tu ejemplo).
SYN_DATA = "src/data/generated_datasets/last_model"

model_types = {
    MNISTModelType.SIMPLE_CNN:  "Simple CNN",
    MNISTModelType.COMPLEX_CNN: "Complex CNN",
    MNISTModelType.RESNET_PREACT: "ResNet PreAct",
}

history = {"model": [], "epoch": [], "accuracy": [], "loss": []}

for model_type, model_name in model_types.items():
    print(f"\n== Entrenando: {model_name} ==")
    trainer = MnistTrainer(
        model_type=model_type,
        num_epochs=30,
        learning_rate=0.001,
        batch_size=64,
        use_synthetic_dataset=True,        # <- cambio clave
        synthetic_data_dir=SYN_DATA        # <- ruta al .gz
    )

    best_val_loss, patience_counter = float("inf"), 0
    val_accuracies, val_losses = [], []

    for epoch in range(1, trainer.num_epochs + 1):
        trainer.train_epoch(epoch)
        val_loss, val_acc = trainer.evaluate_on_loader(
            trainer.val_loader, loader_name="Validación"
        )

        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        if model_type == MNISTModelType.DECISION_TREE:
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            if patience_counter >= trainer.early_stopping_patience:
                break

    for i, (acc, loss) in enumerate(zip(val_accuracies, val_losses)):
        history["model"].append(model_name)
        history["epoch"].append(i + 1)
        history["accuracy"].append(acc)
        history["loss"].append(loss)

df = pd.DataFrame(history)


# Graficar Accuracy
plt.figure(figsize=(10, 5))
for model in df["model"].unique():
    sub = df[df["model"] == model]
    plt.plot(sub["epoch"], sub["accuracy"], label=model)
plt.title("Precisión en Validación por Época")
plt.xlabel("Época")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("val_accuracy_curve.png")
plt.show()

# Graficar Loss
plt.figure(figsize=(10, 5))
for model in df["model"].unique():
    sub = df[df["model"] == model]
    plt.plot(sub["epoch"], sub["loss"], label=model)
plt.title("Pérdida Promedio en Validación por Época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("val_loss_curve.png")
plt.show()
