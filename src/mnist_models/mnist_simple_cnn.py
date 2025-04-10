import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import MNIST_N_CLASSES

class MNISTCNN(nn.Module):
    """
    Modelo de red neuronal convolucional para la clasificación de dígitos MNIST.
    Esta clase se encarga únicamente de definir la arquitectura.
    """
    def __init__(self, num_classes=MNIST_N_CLASSES):
        super(MNISTCNN, self).__init__()
        # Primera capa de convolución: de 1 canal a 32, kernel 3x3.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Segunda capa de convolución: de 32 a 64 canales.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Max pooling con ventana 2x2 para reducir la dimensionalidad.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout para ayudar a reducir el overfitting.
        self.dropout = nn.Dropout(0.25)
        # Capa totalmente conectada. Después de dos pooling, el tamaño 28x28 se reduce a 7x7.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Capa de salida con 'num_classes' nodos (por defecto 10).
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Aplicación de la primera convolución, activación ReLU y pooling.
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # Segunda convolución, ReLU y pooling.
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Aplicación de dropout.
        x = self.dropout(x)
        # Aplanamos para las capas lineales.
        x = x.view(x.size(0), -1)
        # Primera capa lineal seguida de ReLU y dropout.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Capa de salida (la función de pérdida aplicará la softmax si es necesario).
        x = self.fc2(x)
        return x
