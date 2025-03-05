import os
import torch

# Definir rutas generales
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configuración del dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CPU = False

# Configuración del entrenamiento
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS_TEACHER = 40  # Modelo final debería ser 100
EPOCHS_STUDENT = 50
LOG_FREQ = 10
N_SAMPLES_TRAIN = 36
N_SAMPLES_GENERATE = 1000

# Configuración del modelo
MODEL_BASE_DIM = 64
MODEL_DIM_MULTS = [2, 4]
MODEL_IN_CHANNELS = 1
MODEL_IMAGE_SIZE = 28
TIMESTEPS = 1000
MODEL_EMA_STEPS = 10
MODEL_EMA_DECAY = 0.995

# Directorios de datos
DATA_DIR = "src/data/labeled_synthetic"
OUTPUT_LABELED_DIR = "src/data/labeled_synthetic"
SYNTHETIC_DIR = "src/data/synthetic"
MNIST_DATA_LOADERS_DIR = "src/data/mnist_data"

# Directorios de modelos y checkpoints
SAVE_TEACHER_DATA_DIR = "src/data/train/teacher_epochs"
CKTP = ''
MODEL_PATH = "src/data/train/teacher_epochs/model.pt"
