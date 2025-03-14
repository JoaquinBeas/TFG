# models/mnist_student_cnn.py
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from config import MODEL_IMAGE_SIZE, MODEL_IN_CHANNELS, TIMESTEPS

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    def forward(self, x):
        # x: [B, C, H, W] -> [seq_len, B, C]
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).permute(2, 0, 1)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(1, 2, 0).view(B, C, H, W)
        return attn_out

class MNISTStudentCNN(nn.Module):
    def __init__(self, 
                 image_size=MODEL_IMAGE_SIZE, 
                 in_channels=MODEL_IN_CHANNELS, 
                 time_embedding_dim=256, 
                 timesteps=TIMESTEPS,
                 num_heads=4):
        super().__init__()
        # Parámetros para sampling
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
        
        # Encoder CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # [B, 32, 28, 28]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # [B, 64, 14, 14]
        self.attn = SelfAttentionBlock(embed_dim=64, num_heads=num_heads)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # [B, 64, 14, 14]
        
        # Time embedding
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Upsampling y capa final
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 14->28
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=1)
    
    def forward(self, x, noise, t=None):
        out = self.conv1(x)         # [B, 32, 28, 28]
        out = self.conv2(out)         # [B, 64, 14, 14]
        out = self.attn(out)          # [B, 64, 14, 14]
        out = self.conv3(out)         # [B, 64, 14, 14]
        
        if t is not None:
            t_emb = self.time_embedding(t)  # [B, 256]
            t_emb = self.time_mlp(t_emb)      # [B, 64]
            t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
            out = out + t_emb
        
        out = self.upsample(out)      # [B, 64, 28, 28]
        pred_noise = self.final_conv(out)  # [B, 1, 28, 28]
        return pred_noise

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise):
        pred = self.forward(x_t, noise, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred)
        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            std = torch.sqrt(beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod))
        else:
            std = 0.0
        return mean + std * noise

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, t, noise):
        pred = self.forward(x_t, noise, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        x_0_pred = (torch.sqrt(1.0 / alpha_t_cumprod) * x_t - torch.sqrt(1.0 / alpha_t_cumprod - 1.0) * pred)
        x_0_pred.clamp_(-1.0, 1.0)
        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            mean = (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)) * x_0_pred + \
                   ((1.0 - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_t_cumprod)) * x_t
            std = torch.sqrt(beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod))
        else:
            mean = (beta_t / (1.0 - alpha_t_cumprod)) * x_0_pred
            std = 0.0
        return mean + std * noise

    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda"):
        x_t = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size)).to(device)
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i for _ in range(n_samples)]).to(device)
            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise)
            else:
                x_t = self._reverse_diffusion(x_t, t, noise)
        x_t = (x_t + 1.0) / 2.0
        return x_t
