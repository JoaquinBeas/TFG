# models/mnist_student_resnet.py
import torch.nn as nn
import torch.nn.functional as F
from config import *

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.relu(out)
        return out

class MNISTStudentResNet(nn.Module):
    def __init__(self, 
                 image_size=MODEL_IMAGE_SIZE, 
                 in_channels=MODEL_IN_CHANNELS, 
                 time_embedding_dim=256, 
                 timesteps=TIMESTEPS):
        super().__init__()
        # Encoder
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)   # 28x28
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)    # 28 -> 14
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)   # 14 -> 7
        
        # Time embedding
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Decoder: Upsample from 7x7 back to 28x28
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
    
    def forward(self, x, noise, t=None):
        # Similar a teacher, ignoramos 'noise' aquí y nos enfocamos en predecir el ruido.
        out = self.initial(x)        # [B, 64, 28, 28]
        out = self.layer1(out)       # [B, 64, 28, 28]
        out = self.layer2(out)       # [B, 128, 14, 14]
        out = self.layer3(out)       # [B, 256, 7, 7]
        
        if t is not None:
            t_emb = self.time_embedding(t)       # [B, 256]
            t_emb = self.time_mlp(t_emb)           # [B, 256]
            t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
            out = out + t_emb  # Inyección de información temporal
        
        out = self.up1(out)          # [B, 128, 14, 14]
        out = self.up2(out)          # [B, 64, 28, 28]
        pred_noise = self.final_conv(out)  # [B, 1, 28, 28]
        return pred_noise
