import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, output_path):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Curva de Entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
