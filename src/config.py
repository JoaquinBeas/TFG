import os
import torch

# Rutas generales
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configuración del dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CPU = False

# Configuración del entrenamiento
BATCH_SIZE = 128
LEARNING_RATE = 0.002
EPOCHS_TEACHER = 2  # Modelo final debería ser 100
EPOCHS_STUDENT = 2
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
DATA_DIR = os.path.join("src", "data", "labeled_synthetic")
OUTPUT_LABELED_DIR = os.path.join("src", "data", "labeled_synthetic")
SYNTHETIC_DIR = os.path.join("src", "data", "synthetic")
MNIST_DATA_LOADERS_DIR = os.path.join("src", "data", "mnist_data")

# NUEVA estructura para guardar modelos y resultados de entrenamiento

# Carpeta para guardar los modelos (checkpoints .pt)
SAVE_MODELS_DIR = os.path.join("src", "data", "models_pt")
CKTP = ''
# Por ejemplo, para teacher y para student guided
MODEL_PATH = os.path.join(SAVE_MODELS_DIR, "model_teacher.pt")
MODEL_PATH_STUDENT = os.path.join(SAVE_MODELS_DIR, "model_student_guided.pt")

# Carpeta para guardar imágenes de cada época.
# Dentro de "src/data/train" se crearán subcarpetas para cada modelo.
TRAIN_DIR = os.path.join("src", "data", "train")
SAVE_TEACHER_IMAGES_DIR = os.path.join(TRAIN_DIR, "model_teacher_epochs")
SAVE_STUDENT_IMAGES_DIR = os.path.join(TRAIN_DIR, "model_student_epochs")
