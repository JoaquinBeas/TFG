import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from enum import Enum
from src.diffusion_models.diffusion_guided_unet import DiffusionGuidedUnet
from src.diffusion_models.diffusion_resnet import DiffusionResnet
from src.diffusion_models.diffusion_unet import DiffusionUnet
from src.utils.config import DEVICE, MNIST_BATCH_SIZE, MNIST_EPOCHS, MNIST_LEARNING_RATE, TRAIN_DIFFUSION_MODEL_DIR, TRAIN_DIFFUSION_SAMPLES_DIR, MODEL_IMAGE_SIZE, MODEL_IN_CHANNELS, TIMESTEPS
from src.utils.data_loader import get_mnist_dataloaders
from src.utils.mnist_models_enum import DiffusionModelType

class DiffussionTrainer:
    def __init__(
        self,
        model_type: DiffusionModelType = DiffusionModelType.GUIDED_UNET,
        num_epochs=MNIST_EPOCHS,
        learning_rate=MNIST_LEARNING_RATE,
        batch_size=MNIST_BATCH_SIZE,
        early_stopping_patience=3,
        model_path=TRAIN_DIFFUSION_MODEL_DIR,
        image_path=TRAIN_DIFFUSION_SAMPLES_DIR
    ):
        """
        Inicializa el entrenador para el modelo de difusión.
        Se configura el dispositivo, se cargan los dataloaders (se utiliza MNIST como ejemplo),
        se instancia el modelo correspondiente al valor del enum, el optimizador y la función de pérdida (MSE).
        Además, se realiza la división del dataset de entrenamiento en 85%/15% para entrenamiento y validación
        y se crea el directorio para guardar los checkpoints.

        Args:
            model_type (DiffusionModelType): Enum que indica qué modelo de difusión entrenar.
            num_epochs (int): Número de épocas de entrenamiento.
            learning_rate (float): Tasa de aprendizaje para el optimizador.
            batch_size (int): Tamaño de lote para el entrenamiento.
            early_stopping_patience (int): Número de épocas sin mejora en validación antes de detener el entrenamiento.
            model_path (str): Ruta base donde se guardarán los checkpoints.
            image_path (str): Ruta base donde se guardarán las imágenes de muestra.
        """
        self.device = torch.device(DEVICE)
        full_train_loader, self.test_loader = get_mnist_dataloaders(batch_size=batch_size)
        
        # Dividir el dataset de entrenamiento en 85% y 15%
        train_dataset = full_train_loader.dataset
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Seleccionar e instanciar el modelo según el tipo del enum.
        if model_type == DiffusionModelType.GUIDED_UNET:
            self.model = DiffusionGuidedUnet().to(self.device)
        elif model_type == DiffusionModelType.RESNET:
            self.model = DiffusionResnet().to(self.device)
        elif model_type == DiffusionModelType.UNET:
            self.model = DiffusionUnet(image_size=MODEL_IMAGE_SIZE, in_channels=MODEL_IN_CHANNELS, timesteps=TIMESTEPS).to(self.device)
        else:
            raise ValueError("Modelo de difusión desconocido.")
        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()  # Se minimiza el error cuadrático entre el ruido predicho y el real.
        
        # Usar el valor del enum para crear la carpeta de checkpoints.
        self.checkpoint_dir = os.path.join(model_path, model_type.value)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Directorio de checkpoints: {self.checkpoint_dir}")
        
        # Crear una carpeta de imágenes dentro del directorio base, usando el nombre del modelo.
        self.image_path = os.path.join(image_path, model_type.value)
        os.makedirs(self.image_path, exist_ok=True)
        print(f"Directorio de imágenes: {self.image_path}")

        self.early_stopping_patience = early_stopping_patience

    def train_epoch(self, epoch):
        """
        Ejecuta el entrenamiento para una única época.
        
        Para cada batch:
         - Se obtiene el lote de imágenes.
         - Se muestrea un tensor de ruido con la misma forma que las imágenes.
         - Se realiza forward: el modelo selecciona un timestep, genera una imagen ruidosa y predice el ruido.
         - Se calcula la pérdida MSE entre el ruido predicho y el ruido real y se realiza backpropagation.
        
        Args:
            epoch (int): Número de la época actual.
        """
        self.model.train()
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            # Dentro del ciclo de entrenamiento, en lugar de:
            noise = torch.randn_like(data)

            # Se podría hacer:
            if isinstance(self.model, DiffusionResnet):
                batch_size = data.size(0)
                # Usar el feature_dim que se espera (512 por defecto)
                noise = torch.randn(batch_size, 512, device=self.device)
            else:
                noise = torch.randn_like(data)

            self.optimizer.zero_grad()
            predicted_noise = self.model(data, noise)
            loss = self.criterion(predicted_noise, noise)
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Época {epoch} - Batch {batch_idx}/{len(self.train_loader)} - Pérdida: {loss.item():.6f}")
    
    def evaluate_on_loader(self, loader, loader_name=""):
        """
        Evalúa el modelo en el DataLoader especificado (validación o test).
        
        Args:
            loader (DataLoader): DataLoader a evaluar.
            loader_name (str): Nombre descriptivo para el loader.
        
        Returns:
            float: Pérdida promedio (MSE) en el loader evaluado.
        """
        self.model.eval()
        total_loss = 0.0
        total = 0
        
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                noise = torch.randn_like(data)

                # Se podría hacer:
                if isinstance(self.model, DiffusionResnet):
                    batch_size = data.size(0)
                    # Usar el feature_dim que se espera (512 por defecto)
                    noise = torch.randn(batch_size, 512, device=self.device)
                else:
                    noise = torch.randn_like(data)                
                predicted_noise = self.model(data, noise)
                loss = self.criterion(predicted_noise, noise)
                total_loss += loss.item() * data.size(0)
                total += data.size(0)
        
        avg_loss = total_loss / total
        if loader_name:
            print(f"{loader_name} - Pérdida Promedio: {avg_loss:.6f}")
        else:
            print(f"Pérdida Promedio: {avg_loss:.6f}")
        return avg_loss
    
    def train_model(self):
        """
        Ejecuta el ciclo de entrenamiento y validación con early stopping basado en la pérdida de validación.
        
        Tras cada época:
          - Se entrena el modelo.
          - Se evalúa en el conjunto de validación.
          - Se guarda un checkpoint (nombre de la época) en el directorio configurado.
          - Si no mejora la pérdida de validación durante 'early_stopping_patience' épocas consecutivas,
            se detiene el entrenamiento.
        
        Finalmente, se evalúa sobre el conjunto de test y se guarda el modelo final como 'last_model.pt'.
        
        Returns:
            float: Pérdida promedio evaluada sobre el conjunto de test.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n=== Época {epoch} ===")
            self.train_epoch(epoch)
            
            # Evaluar en validación.
            val_loss = self.evaluate_on_loader(self.val_loader, loader_name="Validación")
            
            # Guardar el checkpoint de la época actual.
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{epoch}.pt")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Modelo guardado para la época {epoch} en {checkpoint_path}")
            # Generar y guardar una muestra (1 sample) para la época actual
            sample = self.model.sampling(9, clipped_reverse_diffusion=True, device=self.device)
            sample_path = os.path.join(self.image_path, f"epoch_{epoch}.png")
            save_image(sample, sample_path, nrow=1, normalize=True)
            print(f"Muestra guardada en {sample_path}")
            # Early stopping: si no hay mejora en la pérdida de validación, incrementar el contador.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No hay mejora en la pérdida de validación. Paciencia: {patience_counter}/{self.early_stopping_patience}")
                if patience_counter >= self.early_stopping_patience:
                    print("Early stopping activado. Finalizando entrenamiento.")
                    break
        
        # Evaluar el modelo final sobre el conjunto de test.
        avg_loss_test = self.evaluate_on_loader(self.test_loader, loader_name="Test")
        
        # Guardar el modelo final como 'last_model.pt'.
        last_model_path = os.path.join(self.checkpoint_dir, "last_model.pt")
        torch.save(self.model.state_dict(), last_model_path)
        print(f"Modelo final guardado en {last_model_path}")
        
        return avg_loss_test
    
    def get_model(self):
        """
        Retorna el modelo entrenado para su uso posterior.
        
        Returns:
            nn.Module: El modelo entrenado.
        """
        return self.model
