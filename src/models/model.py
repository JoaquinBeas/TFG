import torch.nn as nn
import torch
import math
from unet import Unet
from tqdm import tqdm


# La idea principal del modelo es aplicar un proceso de difusión (añadir ruido progresivamente) y, luego aprender a revertirlo usando una arquitectura U-Net
class MNISTDiffusion(nn.Module):
    # Entradas:
    #   image_size: Tamaño (ancho/alto) de la imagen.
    #   in_channels: Número de canales de entrada.
    #   time_embedding_dim (opcional, por defecto 256): Dimensión de la incrustación temporal.
    #   timesteps (opcional, por defecto 1000): Número de pasos en la difusión.
    #   base_dim (opcional, por defecto 32): Dimensión base para el modelo U-Net.
    #   dim_mults (opcional, por defecto [1, 2, 4, 8]): Multiplicadores de dimensión en las distintas etapas del U-Net.
    def __init__(
        self,
        image_size,
        in_channels,
        time_embedding_dim=256,
        timesteps=1000,
        base_dim=32,
        dim_mults=[1, 2, 4, 8],
    ):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size

        betas = self._cosine_variance_schedule(timesteps)  # todo Definir las betas

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(
            alphas, dim=-1
        )  # todo Cumprod es el producto acumulado de las betas

        # Registro de buffers, permite que estos tensores se guarden junto con el modelo y se muevan al dispositivo correcto sin ser considerados parámetros entrenables.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # Inicialización del modelo U-Net
        self.model = Unet(
            timesteps, time_embedding_dim, in_channels, in_channels, base_dim, dim_mults
        )

    # Entradas:
    #   x: Imágenes de entrada con forma NCHW.
    #   noise: Tensor de ruido (con la misma forma que x) que se usará en la difusión.
    # Salidas:
    #   pred_noise:El ruido predicho por el modelo U-Net, que se utilizará en el entrenamiento para comparar con el ruido real.
    def forward(self, x, noise):
        # Radndom num entre 0 y timesteps para cada imagen del batch (tengo que mirar el volumen de batch mas eficiente)
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        # Obtiene la imagen en el paso t del proceso de difusión mezcla de la imagen original y el ruido
        x_t = self._forward_diffusion(x, t, noise)
        # Pasa la imagen y t al modelo U-Net para predecir el ruido añadido.
        pred_noise = self.model(x_t, t)
        return pred_noise

    # Entradas:
    #   n_samples: Número de imágenes que se quieren generar.
    # todo   clipped_reverse_diffusion (opcional): Flag que indica si se debe usar la reversión de difusión con clipping (limitar valores) o la versión estándar.
    # Salidas:
    #   x_t: tensor final que representa la imagen generada
    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda"):
        # Se crea un tensor x_t inicial lleno de ruido (normal)
        x_t = torch.randn(
            (n_samples, self.in_channels, self.image_size, self.image_size)
        ).to(device)
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"):
            # Se crea un tensor noise con la misma forma que x_t
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i for _ in range(n_samples)]).to(device)
            # Segun si se usa clipaje se usa una difusion o otra
            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise)
            else:
                x_t = self._reverse_diffusion(x_t, t, noise)
        x_t = (x_t + 1.0) / 2.0  # [-1,1] to [0,1] modificacion del rango
        return x_t

    # Entradas:
    #    timesteps: Número de pasos de la difusión.
    #    todo epsilon (opcional): Pequeño valor para ajustar la función coseno (por defecto 0.008).
    # Salidas:
    #    betas: Tensor de varianzas para cada paso de la difusión.
    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        # Pasos equidistantes entre 0 y timesteps (default 1000)
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        # Se calcula un tensor usando la funcion del coseno
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        # Se calculan las diferentes betas como la diferencia relativa entre pasos
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    # Entradas:
    #     x_0: Imagen original (sin ruido).
    #     t: Tensor de pasos temporales para cada muestra.
    #     noise: Tensor de ruido (de la misma forma que x_0).
    # Salidas:
    #     Imagen ruidosa en el paso t del proceso de difusión.
    def _forward_diffusion(self, x_0, t, noise):
        assert x_0.shape == noise.shape
        return (
            self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * x_0
            + self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(
                x_0.shape[0], 1, 1, 1
            )
            * noise
        )
     
    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise):
        """
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        """
        pred = self.model(x_t, t)

        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(
            x_t.shape[0], 1, 1, 1
        )
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
            -1, t
        ).reshape(x_t.shape[0], 1, 1, 1)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred
        )

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1, 1, 1
            )
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        return mean + std * noise

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, t, noise):
        """
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        """
        pred = self.model(x_t, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(
            x_t.shape[0], 1, 1, 1
        )
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)

        x_0_pred = (
            torch.sqrt(1.0 / alpha_t_cumprod) * x_t
            - torch.sqrt(1.0 / alpha_t_cumprod - 1.0) * pred
        )
        x_0_pred.clamp_(-1.0, 1.0)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1, 1, 1
            )
            mean = (
                beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            ) * x_0_pred + (
                (1.0 - alpha_t_cumprod_prev)
                * torch.sqrt(alpha_t)
                / (1.0 - alpha_t_cumprod)
            ) * x_t

            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            mean = (
                beta_t / (1.0 - alpha_t_cumprod)
            ) * x_0_pred  # alpha_t_cumprod_prev=1 since 0!=1
            std = 0.0

        return mean + std * noise
