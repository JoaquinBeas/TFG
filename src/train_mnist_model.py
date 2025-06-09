import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.utils.config import DEVICE, MNIST_BATCH_SIZE, MNIST_EPOCHS, MNIST_LEARNING_RATE, MNIST_PATIENCE, SAVE_SYNTHETIC_DATASET_DIR, TRAIN_MNIST_MODEL_DIR
# Importamos ambas funciones para cargar los datasets:
from src.utils.data_loader import get_mnist_dataloaders, get_synthetic_mnist_dataloaders
from src.utils.mnist_models_enum import MNISTModelType
from src.utils.config import MNIST_N_CLASSES

class MnistTrainer:
    def __init__( self, model_type: MNISTModelType = MNISTModelType.SIMPLE_CNN, num_epochs=MNIST_EPOCHS, learning_rate=MNIST_LEARNING_RATE, batch_size=MNIST_BATCH_SIZE, early_stopping_patience=MNIST_PATIENCE, model_path=TRAIN_MNIST_MODEL_DIR, use_synthetic_dataset: bool = False, synthetic_data_dir: str | None = None):
        """
        Inicializa el entrenador configurando el dispositivo, los dataloaders,
        el modelo, el optimizador y la función de pérdida.
        Se crea el directorio para guardar los checkpoints del modelo.
        Se realiza una división del conjunto de entrenamiento: 85% para entrenamiento y 15% para validación (early stopping).

        Args:
            model_type (MNISTModelType): Enum que indica qué modelo entrenar.
            num_epochs (int): Número de épocas de entrenamiento.
            learning_rate (float): Tasa de aprendizaje para el optimizador.
            batch_size (int): Tamaño de lote para el entrenamiento.
            early_stopping_patience (int): Número de épocas sin mejora en la pérdida de validación antes de detener el entrenamiento.
            model_path (str): Ruta base para guardar los checkpoints.
            use_synthetic_dataset (bool): Si es True se utilizará el dataset sintético en lugar del dataset original de MNIST.
        """
        self.device = torch.device(DEVICE)
        
        # Cargar el dataset utilizando el método correspondiente
        if use_synthetic_dataset:
            print(f"Cargando dataset sintético desde {synthetic_data_dir or SAVE_SYNTHETIC_DATASET_DIR} …")
            full_train_loader, test_loader = get_synthetic_mnist_dataloaders(batch_size=batch_size,synthetic_data_dir=synthetic_data_dir)
            # full_train_loader.dataset es el dataset completo obtenido por la función get_synthetic_mnist_dataloaders
            full_train_dataset = full_train_loader.dataset
        else:
            print("Cargando dataset MNIST original...")
            full_train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)
            full_train_dataset = full_train_loader.dataset
        
        self.test_loader = test_loader
        
        # Dividir el dataset de entrenamiento en 85% para entrenamiento y 15% para validación
        total_len = len(full_train_dataset)
        train_size = int(0.85 * total_len)
        val_size = total_len - train_size
        train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Seleccionar e instanciar el modelo según el enum
        if model_type == MNISTModelType.SIMPLE_CNN:
            from src.mnist_models.mnist_simple_cnn import MNISTCNN
            self.model = MNISTCNN().to(self.device)
        elif model_type == MNISTModelType.COMPRESSED_CNN:
            from src.mnist_models.mnist_compressed_cnn import MNISTNet1
            self.model = MNISTNet1().to(self.device)
        elif model_type == MNISTModelType.RESNET_PREACT:
            from src.mnist_models.resnet_preact import ResNetPreAct
            # Config mínimo inline:
            class Cfg: pass
            cfg = Cfg(); cfg.model = Cfg()
            cfg.model.in_channels    = 1
            cfg.model.n_classes      = MNIST_N_CLASSES
            cfg.model.base_channels  = 16
            cfg.model.block_type     = 'basic'
            cfg.model.depth          = 20
            cfg.model.remove_first_relu = False
            cfg.model.add_last_bn    = False
            cfg.model.preact_stage   = [True, True, True]
            self.model = ResNetPreAct(cfg).to(self.device)
        elif model_type == MNISTModelType.DECISION_TREE:
            from src.mnist_models.mnist_decision_tree import MNISTDecisionTree
            self.model = MNISTDecisionTree(max_depth=40)  # o lo que quieras
        else:
            raise ValueError("Modelo MNIST desconocido.")
        if model_type == MNISTModelType.DECISION_TREE:
            self.num_epochs = 1
            self.early_stopping_patience = 0
        else:
            self.num_epochs = num_epochs
            self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Crear el directorio para almacenar los checkpoints del modelo basado en el valor del enum
        self.checkpoint_dir = os.path.join(model_path, model_type.value)
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
                print(f"Época {epoch} Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.6f}")
    
    def evaluate_on_loader(self, loader, loader_name="Validation"):
        """
        Evalúa el modelo en el DataLoader especificado (validación o test).

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
        Ejecuta el ciclo completo de entrenamiento y evaluación usando early stopping basado en la pérdida de validación.
        Después de cada época, guarda un checkpoint del modelo. Al finalizar el entrenamiento (por early stopping o completando todas las épocas),
        guarda el modelo final como 'last_model.pt'.

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
