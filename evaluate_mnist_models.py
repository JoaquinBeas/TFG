import os
import torch
from torch.utils.data import DataLoader
from src.utils.config import DEVICE, TRAIN_MNIST_MODEL_DIR, TRAIN_MNIST_MODEL_COPY_DIR, MNIST_BATCH_SIZE
from src.utils.data_loader import get_mnist_dataloaders
from src.train_mnist_model import MNISTModelType
from src.mnist_models.mnist_simple_cnn import MNISTCNN
from src.mnist_models.mnist_complex_cnn import MNISTNet1


def load_model(model_base_dir: str, model_type: MNISTModelType) -> torch.nn.Module:
    """
    Carga el modelo correspondiente (simple o complejo) desde su carpeta 'last_model.pt'.
    """
    # Selección de clase según tipo
    if model_type == MNISTModelType.SIMPLE_CNN:
        model = MNISTCNN()
    elif model_type == MNISTModelType.COMPLEX_CNN:
        model = MNISTNet1()
    else:
        raise ValueError(f"Tipo de modelo MNIST desconocido: {model_type}")

    # Ruta al checkpoint
    ckpt_path = os.path.join(model_base_dir, model_type.value, 'last_model.pt')
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"Cargado {model_type.value} desde {ckpt_path}")
    return model


def evaluate(model: torch.nn.Module, loader: DataLoader):
    """
    Evalúa precisión del modelo sobre un DataLoader.
    Devuelve (accuracy%, preds, labels).
    """
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    accuracy = 100.0 * correct / total
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return accuracy, all_preds, all_labels


def main():
    teacher_type = MNISTModelType.COMPLEX_CNN
    student_type = MNISTModelType.SIMPLE_CNN

    _, test_loader = get_mnist_dataloaders(batch_size=MNIST_BATCH_SIZE)
    teacher = load_model(TRAIN_MNIST_MODEL_DIR, teacher_type)
    student = load_model(TRAIN_MNIST_MODEL_COPY_DIR, student_type)

    # 1) Teacher vs Ground Truth
    t_acc, t_preds, _ = evaluate(teacher, test_loader)
    print(f"Teacher vs GT Accuracy: {t_acc:.2f}%")

    # 2) Student vs Ground Truth
    s_acc, s_preds, _ = evaluate(student, test_loader)
    print(f"Student vs GT Accuracy: {s_acc:.2f}%")

    # 3) Student vs Teacher (acuerdo)
    agree = 100.0 * (s_preds == t_preds).sum().item() / len(t_preds)
    print(f"Agreement (Student vs Teacher): {agree:.2f}%")


if __name__ == '__main__':
    main()
