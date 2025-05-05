import math
import torch


def _cosine_variance_schedule(timesteps=1000, epsilon=0.008):
        # Pasos equidistantes entre 0 y timesteps (default 1000)
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        # Se calcula un tensor usando la funcion del coseno
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        # Se calculan las diferentes betas como la diferencia relativa entre pasos
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas