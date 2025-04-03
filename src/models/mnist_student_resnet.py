# models/mnist_student_resnet.py
import torch
import torch.nn as nn
import math
from tqdm import tqdm

from src.config import MODEL_IMAGE_SIZE, MODEL_IN_CHANNELS, TIMESTEPS
# Descomentar para ejecutar desde aqui
# from config import MODEL_IMAGE_SIZE, MODEL_IN_CHANNELS, TIMESTEPS

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class MNISTStudentResNet(nn.Module):
    def __init__(self, 
                 image_size=MODEL_IMAGE_SIZE, 
                 in_channels=MODEL_IN_CHANNELS, 
                 time_embedding_dim=256, 
                 timesteps=TIMESTEPS):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size
        
        # Inicialización de los parámetros de difusión (igual que en el teacher)
        betas = self._cosine_variance_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # Encoder de la red ResNet (sin downsampling en la primera capa)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # Mantiene 28x28

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)   # 28x28
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)    # 28x28 -> 14x14
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)   # 14x14 -> 7x7
        
        # Time embedding
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Decoder: Upsampling para volver a 28x28
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 7x7 -> 14x14
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 14x14 -> 28x28
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=1)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps+1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    def _forward_diffusion(self, x_0, t, noise):
        # Combina la imagen original con ruido usando los coeficientes del schedule.
        # Asegurarse de que t tenga forma [B] y se adapte a la forma de los buffers.
        factor1 = self.sqrt_alphas_cumprod.gather(0, t).reshape(x_0.shape[0], 1, 1, 1)
        factor2 = self.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(x_0.shape[0], 1, 1, 1)
        return factor1 * x_0 + factor2 * noise

    def forward(self, x, noise, t=None):
        # Si no se pasa t, se genera uno aleatorio
        if t is None:
            t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)
        # Generar la imagen ruidosa usando el proceso de forward diffusion
        x_t = self._forward_diffusion(x, t, noise)
        # Procesamiento de la imagen ruidosa a través de la red
        out = self.initial(x_t)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # Inyecta la información temporal usando el mismo t recibido
        t_emb = self.time_embedding(t)    # [B, 256]
        t_emb = self.time_mlp(t_emb)        # [B, 256]
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
        out = out + t_emb
        # Upsampling para recuperar la resolución original
        out = self.up1(out)
        out = self.up2(out)
        pred_noise = self.final_conv(out)
        return pred_noise

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise):
        # Predice el ruido usando la red (forward) y calcula la reversión.
        pred = self.forward(x_t, noise, t)  # Nota: aquí se usa el forward para predecir el ruido.
        alpha_t = self.alphas.gather(0, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(0, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(0, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(x_t.shape[0], 1, 1, 1)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred)
        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(0, (t - 1)).reshape(x_t.shape[0], 1, 1, 1)
            std = torch.sqrt(beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod))
        else:
            std = 0.0
        return mean + std * noise

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, t, noise):
        pred = self.forward(x_t, noise, t)
        alpha_t = self.alphas.gather(0, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(0, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(0, t).reshape(x_t.shape[0], 1, 1, 1)
        x_0_pred = (torch.sqrt(1.0 / alpha_t_cumprod) * x_t - torch.sqrt(1.0 / alpha_t_cumprod - 1.0) * pred)
        x_0_pred.clamp_(-1.0, 1.0)
        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(0, (t - 1)).reshape(x_t.shape[0], 1, 1, 1)//TODO:
            mean = (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)) * x_0_pred + \
                   ((1.0 - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_t_cumprod)) * x_t
            std = torch.sqrt(beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod))//TODO:
        else:
            mean = (beta_t / (1.0 - alpha_t_cumprod)) * x_0_pred
            std = 0.0
        return mean + std * noise

    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda"):
        # Comienza con un tensor de ruido y aplica el proceso de reverse diffusion.
        x_t = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size)).to(device)
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"):
            noise = torch.randn_like(x_t).to(device) //TODO:
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise)
            else:
                x_t = self._reverse_diffusion(x_t, t, noise)
        # Normaliza la salida al rango [0, 1]
        x_t = (x_t + 1.0) / 2.0
        return x_t