from enum import Enum

class DiffusionModelType(Enum):
    # GUIDED_UNET = "diffusion_guided_unet"
    CONDITIONAL_UNET = "conditional_unet"
    UNET = "diffusion_unet"