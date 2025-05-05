from enum import Enum

class DiffusionModelType(Enum):
    GUIDED_UNET = "diffusion_guided_unet"
    UNET = "diffusion_unet"
    CONDITIONAL_UNET = "conditional_unet"