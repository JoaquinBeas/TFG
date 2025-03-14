from models.mnist_teacher import MNISTDiffusion

class MNISTStudent(MNISTDiffusion):
    def __init__(self, image_size=28, in_channels=1, time_embedding_dim=256, timesteps=1000, base_dim=32, dim_mults=[1, 2, 4, 8]):
        super().__init__(image_size, in_channels, time_embedding_dim, timesteps, base_dim, dim_mults)