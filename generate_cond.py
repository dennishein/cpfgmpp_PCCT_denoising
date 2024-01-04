# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch.distributions import Beta
import pydicom
import lpips
import math
import cv2

# temp
import warnings
warnings.filterwarnings('ignore')

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, conditions, hijack, hijack_ext, forward, weight, uncond_score, pfgmpp, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_max = 380 # fix this 
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    NFE = 0 # sanity chgeck 
    if conditions is not None: # conditional case 
      if hijack > 0:
        x_next = conditions.clone().to(torch.float64).to('cuda:0')
        # Reverse run 
        for i, (t_cur, t_next) in enumerate(zip(t_steps[(num_steps-hijack-1):-2], t_steps[num_steps-hijack:-1])):
          x_cur = x_next
            
          # Increase noise temporarily.
          gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
          t_hat = net.round_sigma(t_cur + gamma * t_cur)
          x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            
          # Euler step.
          if uncond_score:
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
          else:
            denoised = net(torch.cat((conditions,x_hat),1), t_hat, class_labels).to(torch.float64)
            
          d_cur = (x_hat - denoised) / t_hat
          x_next = x_hat + (t_next - t_hat) * d_cur
          NFE += 1

          # Apply 2nd order correction.
          final_step = hijack 
          if i < final_step - 1:
              if uncond_score:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
              else:
                denoised = net(torch.cat((conditions,x_next),1), t_hat, class_labels).to(torch.float64)
              d_prime = (x_next - denoised) / t_next
              x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
              NFE += 1
          
          # Data consistency step
          x_next = x_next*weight+conditions.clone().to(torch.float64).to('cuda:0')*(1-weight)
         
      else: # no hijack 
        if pfgmpp:
          x_next = latents.to(torch.float64) 
        else:
          x_next = latents.to(torch.float64) * t_steps[0]
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            if uncond_score:
              denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            else:
              denoised = net(torch.cat((conditions,x_hat),1), t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            NFE += 1

            # Apply 2nd order correction.
            if i < num_steps - 1:
                if uncond_score:
                  denoised = net(x_next, t_next, class_labels).to(torch.float64)
                else:
                  denoised = net(torch.cat((conditions,x_next),1), t_hat, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                NFE += 1
             
            # Data consistency step
            x_next = x_next*weight+torch.tensor(low_pass(conditions), dtype=torch.float64).to('cuda:0')*(1-weight)        
    else: # unconditional case 
      if pfgmpp:
        x_next = latents.to(torch.float64)
      else:
        x_next = latents.to(torch.float64) * t_steps[0]
      
      for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
          x_cur = x_next

          # Increase noise temporarily.
          gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
          t_hat = net.round_sigma(t_cur + gamma * t_cur)
          x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

          # Euler step.
          denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
          d_cur = (x_hat - denoised) / t_hat
          x_next = x_hat + (t_next - t_hat) * d_cur

          # Apply 2nd order correction.
          if i < num_steps - 1:
              denoised = net(x_next, t_next, class_labels).to(torch.float64)
              d_prime = (x_next - denoised) / t_next
              x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]
        self.seeds = seeds
        self.device = device

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def rand_beta_prime(self, size, N=3072, D=128, **kwargs):
        # sample from beta_prime (N/2, D/2)
        # print(f"N:{N}, D:{D}")
        assert size[0] == len(self.seeds)
        latent_list = []
        beta_gen = Beta(torch.FloatTensor([N / 2.]), torch.FloatTensor([D / 2.]))
        #beta_gen = Beta(torch.FloatTensor([256*256 / 2.]), torch.FloatTensor([D / 2.]))
        for seed in self.seeds:
            torch.manual_seed(seed)
            sample_norm = beta_gen.sample().to(kwargs['device']).double()
            # inverse beta distribution
            inverse_beta = sample_norm / (1-sample_norm)
            #if N < 256 * 256 * 3:
            #    sigma_max = 80
            #else:
            sigma_max = kwargs['sigma_max']
            #print(f'Sigma_max is rand_beta_prime: {sigma_max}')

            sample_norm = torch.sqrt(inverse_beta) * sigma_max * np.sqrt(D)
            gaussian = torch.randn(N).to(sample_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2)
            init_sample = unit_gaussian * sample_norm
            latent_list.append(init_sample.reshape((1, *size[1:])))

        latent = torch.cat(latent_list, dim=0)
        return latent

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Auxiliary functions 
def map_to_zero_one_alt(X, min_v = None, max_v = None, return_vals=False):
  if min_v == None:
      min_val = X.min()
      max_val = X.max()
  else:
      min_val = min_v
      max_val = max_v
  X = (X-min_val)/(max_val-min_val)
  if return_vals:
      return X, min_val, max_val
  else:
      return X
      
def inv_map_to_zero_one_alt(X, min_val, max_val):
  """Inverse data normalizer (from [0,1])"""
  return X*(max_val-min_val)+min_val
  
def load_dcm(dcm,dcm_path):
    ds = pydicom.dcmread(dcm_path+dcm+'.dcm')
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    array = ds.pixel_array-1024
    array[array<=-2000] = - 1000
    ds.RescaleIntercept = 0
    ds.WindowCenter = 35
    ds.WindowWidth = 70
    ds.PixelData = array.tobytes()
    return ds
    
def fix_dicom_index(idx,n_digits):
    while len(idx) < n_digits:
        idx = '0'+idx
    return idx
    
def get_data_scaler(data_centered):
  """Data normalizer. Assume data are always in [0, 1]."""
  if data_centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x
    
def get_data_inverse_scaler(data_centered):
  """Inverse data normalizer."""
  if data_centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x
    
def low_pass(data,cutoff=256):
    data = data.detach().cpu().numpy()
    # generate box
    res = data.shape[-1]
    box = torch.zeros(res,res)
    box[res//2-cutoff:res//2+cutoff,res//2-cutoff:res//2+cutoff] = 1
    box = box.numpy()
    # Fourier transform
    ft = np.fft.ifftshift(data)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    # inv Fourier transform 
    ft *=box
    ift = np.fft.ifftshift(ft)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

# SSIM from https://github.com/GBATZOLIS/conditional_score_diffusion
def ssim(img1, img2):
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def SSIM(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def psnr(gt, img):
    """ PSNR values expected in [0,1]"""
    mse = np.mean(np.square((gt - img)))
    return 20 * np.log10(1) - 10 * np.log10(mse)

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--minmax','saved_minmax',   help='Where the saved minmax is located', metavar='PATH',                type=str, default=None,show_default=True)
@click.option('--data',                    help='Where the condition data is located', metavar='PATH',              type=str, default=None,show_default=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--hijack',                  help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--hijack_ext',              help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--forward',                 help='Enable forward diffusion with hijacked', metavar='BOOL',                    type=bool, default=False, show_default=True)
@click.option('--uncond_score',                 help='Conditional via unconditional score function', metavar='BOOL',                    type=bool, default=False, show_default=True)
@click.option('--weight',                  help='Weight in data consistency step', metavar='FLOAT',                 type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))
@click.option('--aug_dim',             help='additional dimension', metavar='INT',                            type=click.IntRange(min=2), default=None, show_default=True)

def main(network_pkl, data, saved_minmax, hijack, hijack_ext, forward, weight, uncond_score, aug_dim, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    
    loss_fn_alex = lpips.LPIPS(net='alex').to('cuda').to(torch.float64) 
        
    # Set up PFGM++ / EDM
    use_pickle = 0
    pfgmpp = 0
    sigma_max = 380
    if aug_dim is not None:
      #use_pickle = 0
      pfgmpp = 1
      #sigma_max = 380./80 
      
    # Set up conditioning data
    if saved_minmax is not None:
      minmax = torch.load('./datasets_unzipped/'+saved_minmax+'.pt')
    else:
      minmax = torch.zeros(2)
    #print(minmax)
    if data is not None:
      conditions = torch.load('./datasets_unzipped/'+data+'.pt') # temporary 
      if conditions.size(1) == 1:
        conditions = torch.cat((conditions,conditions),1)
      if saved_minmax is not None:
        conditions = map_to_zero_one_alt(conditions,minmax[0],minmax[1])

      n_ch = conditions.size(1)//2
      seeds = [0]*conditions.size(0) # override seeds option and run same for each sample
      conditions_iterator = iter(torch.utils.data.DataLoader(dataset=conditions,batch_size=max_batch_size))
      scaler = get_data_scaler(True)
      get_image_metrics = False
      if 'mayo' in data.split('_'):
        get_image_metrics = True
      elif 'priors' in data.split('_'):
        get_image_metrics = True
    inv_scaler = get_data_inverse_scaler(True)
    
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    if use_pickle:
      with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
        #ckpt_num = int(ckpt_dir[-10:-4])
    else:
        net = torch.load(network_pkl, map_location=torch.device('cpu'))
        net = net['ema'].eval().to(device)
        #ckpt_num = int(ckpt_dir[-9:-3])
        if aug_dim is not None:
          assert net.D == aug_dim
    
    #dist.print0(f'Loading network from "{network_pkl}"...')
    #with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    #    net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Set up outdir
    n_steps = sampler_kwargs['num_steps']
    if sampler_kwargs['S_churn']==0:
      class_diff = 'ODE'
    else:
      class_diff = 'SDE'
    if data is not None:
      outdir = os.path.join(re.sub('.pt','',network_pkl),data,class_diff,f'{n_steps}_{hijack}_{hijack_ext}_{weight}_{forward}_{max_batch_size}')
    else:
      outdir = os.path.join(re.sub('.pt','',network_pkl),'unconditional',class_diff,f'{n_steps}')
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')

    # Loop over batches.
    #dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    count = 1 
    list_count = 1
    lpips_list = []
    ssim_list = []
    psnr_list = []
    sample = torch.tensor([])
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
            
        # Set up conditional data 
        conditions = None 
        res = net.img_resolution
        if data is not None:
          conditions = next(conditions_iterator).to(device).to(torch.float64)
          if saved_minmax is None:
            assert conditions.size(0) == 1 # for now 
            conditions, minmax[0], minmax[1] = map_to_zero_one_alt(conditions,return_vals=True)
          conditions = scaler(conditions)
          res = conditions.size(-1)
          
        # Pick latents and labels.
        N = net.img_channels * res * res
        rnd = StackedRandomGenerator(device, batch_seeds)
        if pfgmpp:
            latents = rnd.rand_beta_prime([batch_size, n_ch, res, res],
                                N=N,
                                D=aug_dim,
                                pfgmpp=pfgmpp,
                                device=device,
                                sigma_max=sigma_max) #if (N > 256 * 256 * 3) else 80)
        else:
            latents = rnd.randn([batch_size, n_ch, res, res],
                                device=device)
         
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        if data is not None:
          images = sampler_fn(net, latents, conditions[:,0:n_ch,:,:], hijack, hijack_ext, forward, weight, uncond_score, pfgmpp, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
        else:
          images = sampler_fn(net, latents, None, hijack, hijack_ext, forward, weight, uncond_score, pfgmpp, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
        
        # Get image metrics 
        if get_image_metrics:
          if data is not None:
              for i in range(images.size(0)):
                lpips_list.append((loss_fn_alex(images[i,0,:,:],conditions[i,1,:,:].to(torch.float64))).detach().cpu().item())   
                ssim_list.append((ssim(inv_scaler(images[i,0,:,:]).detach().cpu().numpy(),inv_scaler(conditions[i,1,:,:].to(torch.float64)).detach().cpu().numpy())).item())  
                psnr_list.append((psnr(inv_scaler(images[i,0,:,:]).detach().cpu().numpy(),inv_scaler(conditions[i,1,:,:].to(torch.float64)).detach().cpu().numpy())).item())  
                #print(f'Slice: {list_count} LPIPS: {lpips_list[list_count-1]} SSIM: {ssim_list[list_count-1]} PSNR: {psnr_list[list_count-1]}')
                list_count += 1
                
        # Convert back to HU
        images = inv_scaler(images)
        images = inv_map_to_zero_one_alt(images,minmax[0],minmax[1])
        if data is not None:
          conditions = inv_scaler(conditions)
          conditions = inv_map_to_zero_one_alt(conditions,minmax[0],minmax[1])

        # Save images.
        #images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        #for seed, image_np in zip(batch_seeds, images_np):
        #    image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
        #    os.makedirs(image_dir, exist_ok=True)
        #    image_path = os.path.join(image_dir, f'{seed:06d}.png')
        #    if image_np.shape[2] == 1:
        #        PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
        #    else:
        #        PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        sample = torch.cat((sample,images.detach().cpu()),dim=0)
       
    # Save images.
    os.makedirs(outdir, exist_ok=True)
    torch.save(sample,outdir+'/sample.pt')
        
    if get_image_metrics:
      lpips_avg = sum(lpips_list)/len(lpips_list)
      ssim_avg = sum(ssim_list)/len(ssim_list)
      psnr_avg = sum(psnr_list)/len(psnr_list)
            
      f = open(outdir+"/output.txt", "a")
      f.seek(0)
      f.truncate()                              
      print('*************************',file=f)
      print('*************************',file=f)
      print(f'Current results. D: {aug_dim} T: {n_steps} tau: {hijack} weight: {weight}',file=f)
      print(f'Mean LPIPS: {lpips_avg}',file=f)
      print(f'Mean SSIM: {ssim_avg}',file=f)
      print(f'Mean PSNR: {psnr_avg}',file=f)
      print('*************************',file=f)
      print('*************************',file=f)
      f.close()
        
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
