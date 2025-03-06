import torch.nn as nn
import torch
import math
from utils.unet import Unet

class MNISTStudent(nn.Module):
    def __init__(self, image_size=28, in_channels=1, time_embedding_dim=256, timesteps=1000, base_dim=32, dim_mults=[1, 2, 4, 8]):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size

        betas = self._cosine_variance_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.model = Unet(timesteps, time_embedding_dim, in_channels, in_channels, base_dim, dim_mults)

    def forward(self, x, noise):
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        x_t = self._forward_diffusion(x, t, noise)
        pred_noise = self.model(x_t, t)
        return pred_noise

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2)
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    def _forward_diffusion(self, x_0, t, noise):
        assert x_0.shape == noise.shape
        return (self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * x_0
                + self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * noise)
