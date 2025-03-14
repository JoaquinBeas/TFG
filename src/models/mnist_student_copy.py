from models.mnist_teacher import MNISTDiffusion
from config import *

class MNISTStudent(MNISTDiffusion):
    def __init__(self, image_size=MODEL_IMAGE_SIZE, in_channels=MODEL_IN_CHANNELS, time_embedding_dim=256, timesteps=TIMESTEPS, base_dim=MODEL_BASE_DIM, dim_mults=MODEL_DIM_MULTS):
        super().__init__(image_size, in_channels, time_embedding_dim, timesteps, base_dim, dim_mults)