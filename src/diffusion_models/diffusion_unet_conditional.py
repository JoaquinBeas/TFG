import torch
import torch.nn as nn
import math
from tqdm import tqdm
from src.utils.config import DEVICE, TIMESTEPS, MODEL_IN_CHANNELS, IMAGE_SIZE, MNIST_N_CLASSES
from src.utils.unet_conditional import ConditionalUnet

class ConditionalDiffusionModel(nn.Module):
    """
    Modelo de difusión Gaussiana condicional por clase, con la misma configuración que el DiffusionUnet original.
    Usa una U-Net condicional (embeddings de tiempo + clase).
    """
    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        in_channels: int = MODEL_IN_CHANNELS,
        
        num_classes: int = MNIST_N_CLASSES,
        timesteps: int = TIMESTEPS,
        time_embedding_dim: int = 256,
        base_dim: int = 32,
        dim_mults: list = [1, 2, 4, 8],
        device: str = DEVICE
    ):
        super().__init__()
        self.device = device
        self.timesteps = timesteps
        self.num_classes = num_classes

        # U-Net condicional
        self.model = ConditionalUnet(
            timesteps=timesteps,
            time_embedding_dim=time_embedding_dim,
            num_classes=num_classes,
            in_channels=in_channels,
            out_channels=in_channels,
            base_dim=base_dim,
            dim_mults=dim_mults
        ).to(device)

        # Schedule de varianzas con coseno (match DiffusionUnet)
        betas = self._cosine_variance_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Buffer para difusión
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    @staticmethod
    def _cosine_variance_schedule(timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self.sqrt_alphas_cumprod.gather(0, t).view(-1, 1, 1, 1) * x_start
            + self.sqrt_one_minus_alphas_cumprod.gather(0, t).view(-1, 1, 1, 1) * noise
        )

    def p_losses(self, x_start: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Seleccionar t uniformemente
        t = torch.randint(0, self.timesteps, (x_start.size(0),), device=self.device).long()
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        # Predecir ruido condicionando en la clase
        pred_noise = self.model(x_t, t, labels)
        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        class_labels: torch.Tensor,
        clipped: bool = True
    ) -> torch.Tensor:
        x_t = torch.randn(
            (n_samples, MODEL_IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
            device=self.device
        )
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling conditional"):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            noise = torch.randn_like(x_t)
            # Predicción del ruido
            pred = self.model(x_t, t, class_labels)
            alpha_t = self.alphas.gather(0, t).view(-1, 1, 1, 1)
            alpha_cumprod_t = self.alphas_cumprod.gather(0, t).view(-1, 1, 1, 1)
            beta_t = self.betas.gather(0, t).view(-1, 1, 1, 1)

            if clipped:
                # Estimate x0 y clip
                x0_pred = (x_t - (1 - alpha_cumprod_t).sqrt() * pred) / alpha_cumprod_t.sqrt()
                x0_pred = x0_pred.clamp(-1.0, 1.0)
                prev_cumprod = (
                    self.alphas_cumprod.gather(0, t - 1)
                    .view(-1, 1, 1, 1)
                    if i > 0 else torch.ones_like(alpha_cumprod_t)
                )
                mean = (
                    beta_t * prev_cumprod.sqrt() / (1 - alpha_cumprod_t) * x0_pred
                    + (1 - prev_cumprod) * alpha_t.sqrt() / (1 - alpha_cumprod_t) * x_t
                )
                var = beta_t * (1 - prev_cumprod) / (1 - alpha_cumprod_t)
                x_t = mean + var.sqrt() * noise
            else:
                mean = (1.0 / alpha_t.sqrt()) * (
                    x_t - (1 - alpha_t) / (1 - alpha_cumprod_t).sqrt() * pred
                )
                x_t = mean + beta_t.sqrt() * noise

        # ——— BINARIZACIÓN ———
        # todo pixel > 0 pasa a 1.0, el resto a 0.0
        x_t = (x_t > 0).float()

        return x_t
