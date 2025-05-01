import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.unet_teacher import TimeMLP

class ConditionalUnet(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        time_embedding_dim: int = 256,
        num_classes: int = 10,
        in_channels: int = 1,
        out_channels: int = 1,
        base_dim: int = 32,
        dim_mults: list = [1, 2, 4, 8],
        resnet_block_groups: int = 8,
    ):
        super().__init__()
        # Save hyperparameters
        self.num_classes = num_classes

        # Embeddings for time steps and class labels
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        self.class_embedding = nn.Embedding(num_classes, time_embedding_dim)
        self.cond_proj = nn.Linear(time_embedding_dim, base_dim)

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # Compute channel dimensions at each U-Net scale
        # dims = [32, 64, 128, 256] for dim_mults [1,2,4,8]
        dims = [base_dim * m for m in dim_mults]
        # Pairs for downsampling: (32->64), (64->128), (128->256)
        in_out = list(zip(dims[:-1], dims[1:]))

        # Downsample blocks
        self.downs = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.downs.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(resnet_block_groups, dim_out),
                nn.SiLU(),
            ))

        # Bottleneck: inject conditioning and apply convolutions
        self.mid_mlp = TimeMLP(time_embedding_dim, dims[-1], dims[-1])
        self.mid_block = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
            nn.GroupNorm(resnet_block_groups, dims[-1]),
            nn.SiLU(),
            nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
        )

        # Upsample blocks: use ConvTranspose2d after concatenating skip
        # up_in_dims = [256, 128, 64], skip_dims = same, up_out_dims = [128,64,32]
        up_in_dims = dims[::-1][:-1]
        skip_dims = dims[::-1][:-1]
        up_out_dims = dims[::-1][1:]
        self.ups = nn.ModuleList()
        for in_dim, skip_dim, out_dim in zip(up_in_dims, skip_dims, up_out_dims):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(in_dim + skip_dim, out_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(resnet_block_groups, out_dim),
                nn.SiLU(),
            ))

        # Final 1x1 conv to produce output channels
        self.final_conv = nn.Conv2d(dims[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B,1,28,28), t,y: (B,)
        # Initial conv
        x = self.init_conv(x)  # -> (B,32,28,28)

        # Create combined conditioning vector
        t_emb = self.time_embedding(t)      # (B,256)
        y_emb = self.class_embedding(y)     # (B,256)
        cond = t_emb + y_emb                # (B,256)
        cond_feat = self.cond_proj(cond)    # (B,32)

        # Downsampling with skip storage
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Bottleneck: inject conditioning via MLP and apply convolutions
        x = self.mid_mlp(x, cond)
        x = self.mid_block(x)

        # Upsampling: iterate through ups and reversed skips
        for up, skip in zip(self.ups, reversed(skips)):
            # Align spatial sizes if needed (handling odd dims)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            # Concatenate along channels
            x = torch.cat((x, skip), dim=1)
            x = up(x)

        # Final output conv
        return self.final_conv(x)
