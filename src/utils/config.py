import os
import re

import torch

# Rutas generales
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configuración del dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CPU = False

# Directorios
MNIST_DATA_LOADERS_DIR = os.path.join("src", "data", "mnist_data")
DATA_DIR = os.path.join("src", "data")
TRAIN_MNIST_MODEL_DIR = os.path.join("src", "data","train_mnist_model")
TRAIN_MNIST_MODEL_COPY_DIR = os.path.join("src", "data","train_mnist_model_copy")
TRAIN_DIFFUSION_MODEL_DIR = os.path.join("src", "data","train_diffusion_model")
TRAIN_DIFFUSION_SAMPLES_DIR = os.path.join("src", "data","train_diffusion_sample")
SAVE_SYNTHETIC_DATASET_DIR = os.path.join("src", "data","dataset")
# Constantes utils
BATCH_SIZE = 128
NUM_WORKERS = 4
IMAGE_SIZE = 28
TIMESTEPS = 1000
MODEL_IMAGE_SIZE = 28

# Configuración del entrenamiento para modelos de clasificacion Mnist
MNIST_BATCH_SIZE = 128
MNIST_LEARNING_RATE = 0.001
MNIST_EPOCHS = 70
MNIST_N_CLASSES = 10
MNIST_PATIENCE = 6

# Configuración del entrenamiento para modelos de Difusion

MODEL_BASE_DIM = 64
MODEL_DIM_MULTS = [2, 4]
MODEL_IN_CHANNELS = 1