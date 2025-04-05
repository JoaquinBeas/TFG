import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, output_path):
    """
    Genera y guarda una gráfica con la evolución de la pérdida de entrenamiento y validación.

    Parámetros:
        train_losses (list of float): Lista de pérdidas de entrenamiento por época.
        val_losses (list of float): Lista de pérdidas de validación por época.
        output_path (str): Ruta completa donde se guardará la gráfica (incluyendo el nombre del archivo, por ejemplo, "training_curve.png").
    """
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
