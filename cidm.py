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


    def q_sample(self, x_start, t, noise=None):# 采样p过程x_t
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

    # ---------------------------------------Inpainting-------------------------------------------------

    @torch.no_grad()
    def p_sample_with_mask(self, model, input, x, keep_mask, t, t_index, clip_denoised=True):

        # x_t
        x_t = self.q_sample(input, t)

        x = (keep_mask * x_t + (1 - keep_mask) * x)

        model_output, model_mean, _, model_log_variance = self.p_mean_variance(model, x, t,
                                                                                clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))

        pred_x0 = self.predict_start_from_noise(x, t, model_output)

        if t_index == 0:
            pred_img = model_mean
        else:
            pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return pred_img, pred_x0.clamp(-1, 1)

    def undo(self, img_after, t):
        beta = self._extract(self.betas, t, img_after.shape)

        return torch.sqrt(1 - beta) * img_after + \
               torch.sqrt(beta) * torch.randn_like(img_after)

  
    @torch.no_grad()
    def p_sample_loop_with_mask(self, model, img, mask, n_class=10, w=2, mode='random',
                                schedule=list(reversed(range(0, 1000)))):  
        device = next(model.parameters()).device

        img = torch.Tensor(img).unsqueeze(0).unsqueeze(0).to(device)
        mask = torch.Tensor(mask).unsqueeze(0).unsqueeze(0).to(device)
        batch_size = img.shape[0]
        if mode == 'random':
            cur_y = torch.randint(0, n_class, (batch_size,)).to(device)
        elif mode == 'all':
            if batch_size % n_class != 0:
                batch_size = n_class
                print('change batch_size to', n_class)
            cur_y = torch.tensor([x for x in range(n_class)] * (batch_size // n_class), dtype=torch.long).to(device)
        else:
            cur_y = torch.ones(batch_size).long().to(device) * int(mode)
        # xs = []
        img_after = torch.randn(img.shape, device=device)

        time_pairs = list(zip(schedule[:-1], schedule[1:]))
        # print(time_pairs)
        for t_last, t_cur in time_pairs:

            t_last_t = torch.full((batch_size,), t_last, device=device, dtype=torch.long)

            if t_cur < t_last:
                img_before = img_after.clone()
           
                img_after, pred_x0 = self.p_sample_with_mask(model, img, img_before, mask, t_last_t, cur_y, w, t_last)
          

            else:
                img_after = self.undo(img_after, t=t_last_t + 1)
               

        return pred_x0.numpy()


    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        img,
        mask,
        image_size=256,
        batch_size=1,
        channels=1,
        ddim_timesteps=100,
        ddim_discr_method="uniform",
        ddim_eta=1.0):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        device = next(model.parameters()).device
        
        # start from pure noise (for each example in the batch)
        sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            x_t = self.q_sample(img, t)
            x = sample_img * (1 - mask) + mask * x_t
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, x.shape)
    
            # 2. predict noise using model
            pred_noise = model(x,t)
            
            # 3. get the predicted x_0
            pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

            sample_img = x_prev

        return sample_img.cpu().numpy()

