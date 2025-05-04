from enum import Enum

class DiffusionModelType(Enum):
    GUIDED_UNET = "diffusion_guided_unet"
    RESNET = "diffusion_resnet"
    UNET = "diffusion_unet"
    CONDITIONAL_UNET = "conditional_unet"