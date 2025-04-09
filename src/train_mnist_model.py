import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.utils.config import DEVICE, MNIST_BATCH_SIZE, MNIST_EPOCHS, MNIST_LEARNING_RATE, MNIST_PATIENCE, TRAIN_MNIST_MODEL_DIR
from src.utils.data_loader import get_mnist_dataloaders
from src.mnist_models.mnist_simple_cnn import MNISTCNN

class MnistTrainer:
    def __init__(self, num_epochs=MNIST_EPOCHS, learning_rate=MNIST_LEARNING_RATE, batch_size=MNIST_BATCH_SIZE,
                 model_name="mnist_cnn", early_stopping_patience=MNIST_PATIENCE):
        """
        Inicializa el entrenador configurando el dispositivo, los dataloaders,
        el modelo, el optimizador y la función de pérdida.
        
        Además, crea el directorio para guardar los checkpoints del modelo.
        Se realiza una división del conjunto de entrenamiento, usando 85% para
        entrenamiento y 15% para validación (early stopping).
        
        Args:
            num_epochs (int): Número de épocas de entrenamiento.
            learning_rate (float): Tasa de aprendizaje para el optimizador.
            batch_size (int): Tamaño de lote para el entrenamiento.
            model_name (str): Nombre del modelo, usado para crear la carpeta de checkpoints.
            early_stopping_patience (int): Número de épocas sin mejora en la pérdida de validación
                                           antes de detener el entrenamiento.
        """
        self.device = torch.device(DEVICE)
        
        # Obtener el dataloader completo para entrenamiento y el test_loader
        full_train_loader, self.test_loader = get_mnist_dataloaders(batch_size=batch_size)
        
        # Se extrae el dataset completo de entrenamiento para crear la división (85% train, 15% validación)
        train_dataset = full_train_loader.dataset
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        self.model = MNISTCNN().to(self.device)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = model_name
        self.early_stopping_patience = early_stopping_patience
        
        # Crear el directorio para almacenar los checkpoints del modelo
        self.checkpoint_dir = os.path.join(TRAIN_MNIST_MODEL_DIR, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Directorio de checkpoints: {self.checkpoint_dir}")
    
    def train_epoch(self, epoch):
        """
        Ejecuta el entrenamiento del modelo para una única época.
        
        Args:
            epoch (int): Número de la época actual (para visualización).
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.6f}")
    
    def evaluate_on_loader(self, loader, loader_name="Validation"):
        """
        Evalúa el modelo en el loader que se especifique (puede ser el de validación o test).
        
        Args:
            loader (DataLoader): DataLoader a evaluar.
            loader_name (str): Nombre descriptivo para el loader.
            
        Returns:
            tuple: (avg_loss, accuracy) para el loader evaluado.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                total_loss += loss.item() * data.size(0)
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        print(f"{loader_name} - Pérdida Promedio: {avg_loss:.4f}, Precisión: {accuracy:.2f}%")
        return avg_loss, accuracy

    def train_model(self):
        """
        Ejecuta el ciclo completo de entrenamiento y evaluación usando early stopping basado en
        la pérdida de validación. Después de cada época, guarda un checkpoint del modelo y,
        al finalizar el entrenamiento (por early stopping o completando las épocas), guarda
        el modelo final como 'last_model.pt'.
        
        Returns:
            tuple: (avg_loss_test, accuracy_test) evaluados sobre el conjunto de test.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n=== Época {epoch} ===")
            self.train_epoch(epoch)
            
            # Evaluar en el conjunto de validación
            val_loss, _ = self.evaluate_on_loader(self.val_loader, loader_name="Validación")
            
            # Guardar checkpoint para la época actual
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{epoch}.pt")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Modelo guardado para la época {epoch} en {checkpoint_path}")
            
            # Early Stopping: Si no hay mejora en la pérdida de validación, incrementar el contador
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No mejora en la pérdida de validación. Contador de paciencia: {patience_counter}/{self.early_stopping_patience}")
                if patience_counter >= self.early_stopping_patience:
                    print("Early stopping activado. Finalizando entrenamiento.")
                    break

        # Evaluar el modelo final sobre el conjunto de test
        avg_loss_test, accuracy_test = self.evaluate_on_loader(self.test_loader, loader_name="Test")
        
        # Guardar el modelo final como 'last_model.pt'
        last_model_path = os.path.join(self.checkpoint_dir, "last_model.pt")
        torch.save(self.model.state_dict(), last_model_path)
        print(f"Modelo final guardado en {last_model_path}")
        
        return avg_loss_test, accuracy_test

    def get_model(self):
        """
        Retorna el modelo entrenado para su uso posterior.
        
        Returns:
            nn.Module: El modelo entrenado.
        """
        return self.model
