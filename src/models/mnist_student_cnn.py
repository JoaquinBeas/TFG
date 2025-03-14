# models/mnist_student_cnn.py
import torch.nn as nn
from config import *

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    def forward(self, x):
        # x: [B, C, H, W] => reestructuramos a [seq_len, B, C]
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).permute(2, 0, 1)  # [seq_len, B, C]
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
        
        # Upsampling para volver a 28x28
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=1)  # [B, 1, 28, 28]
    
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
