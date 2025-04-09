import os
import torch
from src.dataset_generator import DataSetGenerator
from src.train_diffussion_model import DiffussionTrainer
from src.train_mnist_model import MnistTrainer
from src.utils.config import DEVICE, TRAIN_DIFUSSION_MODEL_DIR, TRAIN_MNIST_MODEL_DIR

def main():
    train_mnist = False       # True: entrena el modelo MNIST; False: carga el modelo guardado.
    train_diffusion = False   # True: entrena el modelo de difusión; False: carga el modelo guardado.
    mnist_model_name = "mnist_cnn"
    difussion_model_name = 'difussion_guided_unet'
    # ----- Modelo MNIST -----
    if train_mnist:
        print("Entrenando modelo MNIST...")
        trainer_mnist = MnistTrainer(num_epochs=10, learning_rate=0.002, batch_size=64,model_name=mnist_model_name)
        avg_loss, accuracy = trainer_mnist.train_model()
        print(f"Resultados de Evaluación MNIST: Pérdida Promedio = {avg_loss:.4f}, Precisión = {accuracy:.2f}%")
        model_mnist = trainer_mnist.get_model()
    else:
        print("Cargando modelo MNIST desde checkpoint...")
        from src.mnist_models.mnist_simple_cnn import MNISTCNN
        model_mnist = MNISTCNN()
        checkpoint_path = os.path.join(TRAIN_MNIST_MODEL_DIR,mnist_model_name, "last_model.pt")
        model_mnist.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(DEVICE)))
        model_mnist.eval()
        print(f"Modelo MNIST cargado desde: {checkpoint_path}")
        
    # ----- Modelo de Difusión -----
    if train_diffusion:
        print("Entrenando modelo de difusión...")
        trainer_diffusion = DiffussionTrainer(
            num_epochs=10,               
            learning_rate=0.002,         
            batch_size=64,               
            model_name="difussion_guided_unet",
            early_stopping_patience=3
        )
        avg_loss_test = trainer_diffusion.train_model()
        print(f"Entrenamiento de difusión finalizado. Pérdida en Test: {avg_loss_test:.6f}")
        model_diffusion = trainer_diffusion.get_model()
    else:
        print("Cargando modelo de difusión desde checkpoint...")
        from src.difussion_models.difussion_guided_unet import DifussionGuidedUnet
        model_diffusion = DifussionGuidedUnet()
        diffusion_checkpoint_path = os.path.join(TRAIN_DIFUSSION_MODEL_DIR,difussion_model_name, "last_model.pt")
        model_diffusion.load_state_dict(torch.load(diffusion_checkpoint_path, map_location=torch.device(DEVICE)))
        model_diffusion.eval()
        print(f"Modelo de difusión cargado desde: {diffusion_checkpoint_path}")
        
    generator = DataSetGenerator(diffusion_model=model_diffusion, mnist_model=model_mnist)
    generator.generate_dataset(n_samples=100, output_path="generated_dataset.pt")
    
if __name__ == "__main__":
    main()
