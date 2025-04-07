# src/models/mnist_teacher_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import MODEL_IMAGE_SIZE, MODEL_IN_CHANNELS, TIMESTEPS
from src.utils.resnet import resnet18
from src.utils.diffresnet import DiffusionLayer

class MNISTStudentResNet(nn.Module):
    def __init__(self, feature_dim=512, step_size=0.1, diffusion_steps=5):
        """
        feature_dim: dimensión de la representación extraída por la ResNet (por ejemplo, 512).
        step_size: factor de paso para la capa de difusión.
        diffusion_steps: número de veces que se aplica la operación de difusión en el espacio latente.
        """
        super().__init__()
        self.timesteps = TIMESTEPS

        # Precomputación de betas y derivados (igual que en MNISTDiffusion)
        self.register_buffer("betas", self._cosine_variance_schedule(self.timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=-1)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Backbone ResNet; en este ejemplo usamos resnet18.
        # Nota: Si tus imágenes son de 1 canal, adapta la primera capa de resnet18 o replica el canal.
        self.backbone = resnet18(num_classes=feature_dim)
        
        # Capas adicionales para transformar la representación extraída.
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        
        # Parámetro para la difusión: weight_matrix para calcular el laplaciano.
        # Se aprende durante el entrenamiento.
        self.weight_matrix = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.step_size = step_size
        self.diffusion_steps = diffusion_steps
        
        # Capa final para predecir el ruido (la salida tiene la misma dimensión que la representación)
        self.noise_predictor = nn.Linear(feature_dim, feature_dim)
        
        # Decoder sencillo para mapear la representación latente a imagen (MNIST: 28x28 = 784)
        self.decoder = nn.Linear(feature_dim, MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE)

    def compute_laplacian(self):
        weight = self.weight_matrix
        diagonal = torch.diag(weight.sum(dim=1))
        laplacian = diagonal - weight
        return laplacian

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * 0.5 * torch.pi) ** 2
        betas = torch.clamp(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    def _forward_diffusion(self, x0, t, noise):
        """
        Calcula x_t = sqrt(alphas_cumprod[t])*x0 + sqrt(1 - alphas_cumprod[t])*noise.
        Se asume que x0 es la representación extraída (vector de dimensión feature_dim).
        """
        # t es un tensor de enteros [B] que indica el paso para cada muestra.
        sqrt_ac = self.sqrt_alphas_cumprod.gather(-1, t.view(-1, 1)).view(-1, 1)
        sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod.gather(-1, t.view(-1, 1)).view(-1, 1)
        return sqrt_ac * x0 + sqrt_one_minus_ac * noise

    def forward(self, x, noise):
        """
        x: imágenes de entrada.
        noise: tensor de ruido del mismo tamaño que la representación extraída.
        El método:
          1. Extrae características de x mediante la ResNet.
          2. Selecciona un timestep aleatorio para cada muestra y calcula la versión ruidosa x_t.
          3. Pasa x_t por dos capas fully-connected con activación ReLU.
          4. Aplica iterativamente la operación de difusión usando DiffusionLayer (calculado a partir del laplaciano de weight_matrix).
          5. Predice el ruido mediante noise_predictor.
        """
        B = x.size(0)
        device = x.device
        # Extraer representación usando ResNet. Se asume que la función retorna (feature, logits) si se usa return_feature=True.
        feat = self.backbone(x, return_feature=True)[0]  # feat: [B, feature_dim]
        # Seleccionar timesteps aleatorios para cada muestra.
        t = torch.randint(0, self.timesteps, (B,), device=device)
        # Aplicar forward diffusion en el espacio latente.
        latent = self._forward_diffusion(feat, t, noise)
        # Transformación adicional.
        latent = self.fc2(F.relu(self.fc1(latent))) + latent

        # Aplicar iterativamente la operación de difusión.
        laplacian = self.compute_laplacian().to(device)
        for _ in range(self.diffusion_steps):
            # La operación de difusión: x = x - step_size * laplacian * x
            latent = latent - self.step_size * torch.matmul(latent, laplacian.T)
        
        # Predecir el ruido en el espacio latente.
        pred_noise = self.noise_predictor(latent)
        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda"):
        """
        Genera imágenes a partir de ruido puro en el espacio latente mediante reverse diffusion.
        La rutina es similar a la del modelo teacher original:
         - Se inicia con un vector latente aleatorio.
         - Se itera desde timesteps-1 hasta 0 usando la función predictora de ruido.
         - Finalmente, se decodifica el vector latente a una imagen (usando self.decoder).
        """
        B = n_samples
        latent = torch.randn(B, self.backbone.fc.in_features, device=device)
        for i in range(self.timesteps - 1, -1, -1):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            noise = torch.randn_like(latent)
            # Obtener predicción de ruido en el estado actual.
            pred_noise = self.noise_predictor(latent)
            # Extraer valores necesarios del schedule.
            alpha_t = self.alphas.gather(-1, t.view(-1, 1)).sqrt()
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t.view(-1, 1))
            # Reverse diffusion: aproximación simple.
            latent = (1.0 / alpha_t) * (latent - ((1 - self.alphas.gather(-1, t.view(-1, 1))) / sqrt_one_minus_alpha_t) * pred_noise)
            # Si no es el último paso, agregar ruido.
            if i > 0:
                latent = latent + noise
        # Decodificar el vector latente a imagen (MNIST: 28x28, 1 canal).
        images = self.decoder(latent)
        images = images.view(-1, 1, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
        # Ajuste final para estar en el rango [0,1]
        images = (images + 1) / 2.0
        return images
