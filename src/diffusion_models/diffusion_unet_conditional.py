from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from src.utils.config import DEVICE, IMAGE_SIZE, MNIST_N_CLASSES
from src.utils.unet_conditional import ContextUnet, ddpm_schedules

class ConditionalDiffusionModel(nn.Module):
    """
    Conditional DDPM with classifier-free guidance and final binarization
    """
    def __init__(self, nn_model = ContextUnet(in_channels=1, n_feat=256, n_classes=MNIST_N_CLASSES), betas = (1e-4, 0.02), n_T= 400, device= DEVICE, drop_prob= 0.1, num_classes = MNIST_N_CLASSES):
        super(ConditionalDiffusionModel, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.num_classes = num_classes
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))
    
    def p_losses(self, x, c):
        return self.forward(x, c)

    def sample(self, labels, n_sample, size=(1, 28, 28), device=DEVICE, guide_w = 0.5):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        with torch.no_grad():
            x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
            c_i = labels.to(device)                          # shape: (n_sample,)
            context_mask = torch.zeros_like(c_i, dtype=torch.float32, device=device)

            # double the batch
            c_i = c_i.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 1.0 # makes second half of batch context free

            x_i_store = [] # keep track of generated steps in case want to plot something 
            print()
            for i in range(self.n_T, 0, -1):
                print(f'sampling timestep {i}',end='\r')
                t_is = torch.tensor([i / self.n_T]).to(device)
                t_is = t_is.repeat(n_sample,1,1,1)

                # double batch
                x_i = x_i.repeat(2,1,1,1)
                t_is = t_is.repeat(2,1,1,1)

                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

                # split predictions and compute weighting
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1+guide_w)*eps1 - guide_w*eps2
                x_i = x_i[:n_sample]
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
                if i % 20 == 0 or i == self.n_T or i < 8:
                    xi_cpu = x_i.detach().cpu().numpy()
                    x_i_store.append(xi_cpu)
                    del xi_cpu
            torch.cuda.empty_cache()
            x_i_store = np.array(x_i_store)
            return x_i, x_i_store
    @torch.no_grad()
    def sampling(self,
             n_samples: int,
             labels: Optional[torch.LongTensor] = None,
             device: str = DEVICE,
             guide_w: float = 0.5,
             x_t_scale: float = 1.0,
             noise_scale: float = 1.0) -> torch.Tensor:
        """
        Reverse‐diffusion sampler con Classifier‐Free Guidance.
        - n_samples: número de imágenes a generar.
        - labels: LongTensor (n_samples,) con valores [0, num_classes). Si es None, se generan aleatorias.
        - guide_w: fuerza del guidance (w ≥ 0).
        - x_t_scale: escala del ruido inicial x_T.
        - noise_scale: escala del ruido añadido en cada paso (si t > 1).
        Retorna: Tensor (n_samples, 1, 28, 28) con valores en [0,1].
        """
        self.nn_model.eval()

        # — Labels: aleatorias si no se pasan —
        if labels is None:
            labels = torch.randint(0,
                                self.num_classes,
                                (n_samples,),
                                device=device,
                                dtype=torch.long)
        else:
            labels = labels.to(device)

        # — Ruido inicial en x_T —
        x = torch.randn(n_samples, *(1,28,28), device=device) * x_t_scale

        # — Reverse diffusion paso a paso —
        for t in range(self.n_T, 0, -1):
            # embedding de timestep normalizado a [0,1]
            t_norm = torch.full((n_samples,1,1,1),
                                float(t) / self.n_T,
                                device=device)

            # máscaras para guidance
            mask_ctx = torch.zeros(n_samples, device=device)
            mask_noc = torch.ones(n_samples,  device=device)

            # 1) predecir ruido con contexto y sin contexto
            eps_ctx   = self.nn_model(x, labels, t_norm, mask_ctx)
            eps_noctx = self.nn_model(x, labels, t_norm, mask_noc)

            # 2) mezclar según guide_w
            eps = (1 + guide_w) * eps_ctx - guide_w * eps_noctx

            # 3) actualizar x_{t-1} según DDPM:
            #    x_{t-1} = 1/√α_t ( x_t – (1–α_t)/√(1–ᾱ_t) · ε ) + √β_t · z
            inv_sqrt_alpha   = self.oneover_sqrta[t].view(1,1,1,1)
            coef             = self.mab_over_sqrtmab[t].view(1,1,1,1)
            beta_sqrt        = self.sqrt_beta_t[t].view(1,1,1,1)

            x = inv_sqrt_alpha * (x - coef * eps)

            # añadir ruido extra si no es el último paso
            if t > 1:
                x = x + beta_sqrt * torch.randn_like(x, device=device) * noise_scale

        # Mapear de [–1,1] a [0,1]
        images = (x.clamp(-1,1) + 1) * 0.5
        return images