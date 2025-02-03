import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def linear_beta_schedule(timesteps):
    """
    beta schedule
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)



def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)




class GaussianDiffusion:
    def __init__(
        self,
        timesteps=500,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')


        self.betas = betas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) #alphas_t
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.) # alphas_t-1
        self.sqrt_recip_alphas = torch.sqrt(1.0/self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def _extract(self, a, t, x_shape): 
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out


    def q_sample(self, x_start, t, noise=None):# samplex_t
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)
       
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
      
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

  
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    @torch.no_grad()
    def p_sample(self, model, x, t,index):
        betas_t=self._extract(self.betas,t,x.shape)
        sqrt_one_minus_alphas_cumprod_t =self._extract(self.sqrt_one_minus_alphas_cumprod,t,x.shape)
        sqrt_recip_alphas_t=self._extract(self.sqrt_recip_alphas,t,x.shape)
        x_in=x
        model_mean=sqrt_recip_alphas_t*(x-betas_t*model(x_in,t)/sqrt_one_minus_alphas_cumprod_t)

        if index==0:
            return model_mean
        else:
            posterior_variance_t=self._extract(self.posterior_variance,t,x.shape)
            noise=torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t)*noise


    @torch.no_grad()
    def p_sample_loop(self, model,shape):
        # denoise: reverse diffusion
        batch_size = shape[0]
   
        device = 'cuda'
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)  
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model,img, torch.full((batch_size,), i, device=device, dtype=torch.long),i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, x_pred,image_size, batch_size=8, channels=1):
        # sample new images   
        return self.p_sample_loop(model, x_pred,shape=(batch_size, channels, image_size, image_size))


# --------------------------------------------Loss--------------------------------------------

    # compute train losses
    def train_losses(self, model,x_start,t):
        # generate random noise
        noise = torch.randn_like(x_start)

        # get x_t
        x_t = self.q_sample(x_start, t, noise=noise)
      
        predicted_noise = model(x_t, t)

        loss_1 = F.mse_loss(noise, predicted_noise)
        
        # pre_x0 = self.predict_start_from_noise(x_t,t,predicted_noise)
        # loss_2 = F.mse_loss(pre_x0,x_start)
      
        loss = loss_1
      
        return loss

   
