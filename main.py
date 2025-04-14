import os
import torch

from src.synthetic_dataset import SyntheticDataset
from src.train_diffussion_model import DiffussionTrainer, DiffusionModelType
from src.train_mnist_model import MnistTrainer, MNISTModelType
from src.utils.config import DEVICE, TRAIN_DIFFUSION_MODEL_DIR, TRAIN_MNIST_MODEL_COPY_DIR, TRAIN_MNIST_MODEL_DIR

def main():
    # Variables de configuración
    train_mnist = False       # True: entrena el modelo MNIST; False: carga el modelo guardado.
    train_diffusion = False   # True: entrena el modelo de difusión; False: carga el modelo guardado.
    train_diffusion_copy = True   # True: entrena el modelo de difusión; False: carga el modelo guardado.
    
    mnist_model_name = "mnist_complex_cnn"          # Opciones: "mnist_cnn" (modelo simple) o "mnist_complex_cnn" (modelo complejo)
    diffusion_model_name = "diffusion_unet"         # Opciones: "diffusion_guided_unet", "diffusion_resnet" o "diffusion_unet"
    mnist_model_name_copy = "mnist_cnn"             # Opciones: "mnist_cnn" (modelo simple) o "mnist_complex_cnn" (modelo complejo)

    # Seleccionar el modelo MNIST basándonos en la variable definida.
    if mnist_model_name.lower() == "mnist_cnn":
        selected_mnist_model = MNISTModelType.SIMPLE_CNN
    elif mnist_model_name.lower() == "mnist_complex_cnn":
        selected_mnist_model = MNISTModelType.COMPLEX_CNN
    else:
        raise ValueError("Nombre de modelo MNIST desconocido.")

    # Seleccionar el modelo de difusión basándonos en la variable definida.
    if diffusion_model_name.lower() == "diffusion_guided_unet":
        selected_diffusion_model = DiffusionModelType.GUIDED_UNET
    elif diffusion_model_name.lower() == "diffusion_resnet":
        selected_diffusion_model = DiffusionModelType.RESNET
    elif diffusion_model_name.lower() == "diffusion_unet":
        selected_diffusion_model = DiffusionModelType.UNET
    else:
        raise ValueError("Nombre de modelo de difusión desconocido.")

    # Seleccionar el modelo MNIST copia basándonos en la variable definida.
    if mnist_model_name_copy.lower() == "mnist_cnn":
        selected_mnist_model_copy = MNISTModelType.SIMPLE_CNN
    elif mnist_model_name_copy.lower() == "mnist_complex_cnn":
        selected_mnist_model_copy = MNISTModelType.COMPLEX_CNN
    else:
        raise ValueError("Nombre de modelo MNIST desconocido.")

    # ----- MODELO MNIST ----- Teacher
    if train_mnist:
        print("Entrenando modelo MNIST...")
        trainer_mnist = MnistTrainer(
            model_type=selected_mnist_model,
            num_epochs=20,
            learning_rate=0.002,
            batch_size=64
        )
        avg_loss, accuracy = trainer_mnist.train_model()
        print(f"Resultados de Evaluación MNIST: Pérdida Promedio = {avg_loss:.4f}, Precisión = {accuracy:.2f}%")
        model_mnist = trainer_mnist.get_model()
    else:
        print("Cargando modelo MNIST desde checkpoint...")
        if selected_mnist_model == MNISTModelType.SIMPLE_CNN:
            from src.mnist_models.mnist_simple_cnn import MNISTCNN
            model_mnist = MNISTCNN()
        else:
            from src.mnist_models.mnist_complex_cnn import MNISTNet1
            model_mnist = MNISTNet1()
        checkpoint_path = os.path.join(TRAIN_MNIST_MODEL_DIR, selected_mnist_model.value, "last_model.pt")
        model_mnist.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(DEVICE)))
        model_mnist.eval()
        print(f"Modelo MNIST cargado desde: {checkpoint_path}")

    # ----- MODELO DE DIFUSIÓN -----
    if train_diffusion:
        print("Entrenando modelo de difusión...")
        trainer_diffusion = DiffussionTrainer(
            model_type=selected_diffusion_model,
            num_epochs=120,
            learning_rate=0.001,
            batch_size=64,
            early_stopping_patience=100
        )
        avg_loss_test = trainer_diffusion.train_model()
        print(f"Entrenamiento de difusión finalizado. Pérdida en Test: {avg_loss_test:.6f}")
        model_diffusion = trainer_diffusion.get_model()
    else:
        print("Cargando modelo de difusión desde checkpoint...")
        if selected_diffusion_model == DiffusionModelType.GUIDED_UNET:
            from src.diffusion_models.diffusion_guided_unet import DiffusionGuidedUnet
            model_diffusion = DiffusionGuidedUnet()
        elif selected_diffusion_model == DiffusionModelType.RESNET:
            from src.diffusion_models.diffusion_resnet import DiffusionResnet
            model_diffusion = DiffusionResnet()
        elif selected_diffusion_model == DiffusionModelType.UNET:
            from src.diffusion_models.diffusion_unet import DiffusionUnet
            from src.utils.config import MODEL_IMAGE_SIZE, MODEL_IN_CHANNELS, TIMESTEPS
            model_diffusion = DiffusionUnet(
                image_size=MODEL_IMAGE_SIZE,
                in_channels=MODEL_IN_CHANNELS,
                timesteps=TIMESTEPS
            )
        diffusion_checkpoint_path = os.path.join(TRAIN_DIFFUSION_MODEL_DIR, selected_diffusion_model.value, "last_model.pt")
        model_diffusion.load_state_dict(torch.load(diffusion_checkpoint_path, map_location=torch.device(DEVICE)))
        model_diffusion.eval()
        print(f"Modelo de difusión cargado desde: {diffusion_checkpoint_path}")
    print("Generando dataset sintético...")

    # Crear una instancia de SyntheticDataset utilizando el modelo de difusión y el teacher MNIST.
    synthetic_dataset_generator = SyntheticDataset(model_diffusion, model_mnist)

    # Generar, por ejemplo, 1000 muestras; ajusta el número de muestras según tus necesidades.
    synthetic_dataset = synthetic_dataset_generator.generate_dataset(n_samples=1000)
    # ----- MODELO MNIST ----- Student
    if train_diffusion_copy:
        print("Entrenando modelo MNIST...")
        trainer_mnist = MnistTrainer(
            model_type=selected_mnist_model_copy,
            num_epochs=20,
            learning_rate=0.002,
            batch_size=64,
            model_path=TRAIN_MNIST_MODEL_COPY_DIR,
            use_synthetic_dataset=True
        )
        avg_loss, accuracy = trainer_mnist.train_model()
        print(f"Resultados de Evaluación MNIST: Pérdida Promedio = {avg_loss:.4f}, Precisión = {accuracy:.2f}%")
        model_mnist = trainer_mnist.get_model()
    else:
        print("Cargando modelo MNIST desde checkpoint...")
        if selected_mnist_model_copy == MNISTModelType.SIMPLE_CNN:
            from src.mnist_models.mnist_simple_cnn import MNISTCNN
            model_mnist = MNISTCNN()
        else:
            from src.mnist_models.mnist_complex_cnn import MNISTNet1
            model_mnist = MNISTNet1()
        checkpoint_path = os.path.join(TRAIN_MNIST_MODEL_COPY_DIR, selected_mnist_model_copy.value, "last_model.pt")
        model_mnist.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(DEVICE)))
        model_mnist.eval()
        print(f"Modelo MNIST cargado desde: {checkpoint_path}")
if __name__ == "__main__":
    main()
