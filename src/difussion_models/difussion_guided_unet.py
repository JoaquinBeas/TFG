import torch as th
import torch.nn as nn
from src.utils.config import *
from src.utils.unet_guided import UNetModel
from src.utils.gaussian_diffusion import *

class DifussionGuidedUnet(nn.Module):
    def __init__(
        self,
        image_size=MODEL_IMAGE_SIZE,
        in_channels=MODEL_IN_CHANNELS,
        timesteps=TIMESTEPS,
        model_channels=MODEL_BASE_DIM,  # using your config base dimension
        out_channels=MODEL_IN_CHANNELS,
        num_res_blocks=2,
        attention_resolutions=[4],  # Adjust as needed
        dropout=0,
        channel_mult=MODEL_DIM_MULTS,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        **kwargs  # To accept extra unused parameters
    ):
        super().__init__()
        self.timesteps = timesteps
        # Instantiate the guided UNet
        self.unet = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
        # Precompute the beta schedule once.
        betas = get_named_beta_schedule("cosine", self.timesteps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,   # assuming your model predicts epsilon
            model_var_type=ModelVarType.FIXED_SMALL,   # adjust as needed
            loss_type=LossType.MSE,
            rescale_timesteps=True,
        )

    def forward(self, x, noise):
        """
        x: input images (shape [B, C, H, W])
        noise: noise tensor (shape matching x)
        
        This method:
         1. Samples a random timestep t for each image.
         2. Uses the diffusion q_sample to create a noisy image x_t.
         3. Passes x_t and t to the guided UNet.
        """
        B = x.shape[0]
        # Sample random timesteps for each image in the batch.
        t = th.randint(0, self.timesteps, (B,), device=x.device)
        # Generate noisy images using the diffusion process.
        # q_sample takes (x_start, t, noise) and returns x_t.
        x_t = self.diffusion.q_sample(x, t, noise=noise)
        # Now call the guided UNet.
        # The guided UNet expects x_t and a 1D tensor t.
        return self.unet(x_t, t)
    
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda", x_t_scale=1.0, noise_scale=1.0):
        """
        Use the diffusion sampling routine to generate images.
        """
        shape = (n_samples, self.unet.in_channels, self.unet.image_size, self.unet.image_size)
        samples = self.diffusion.p_sample_loop(self.unet, shape, device=device)
        return samples