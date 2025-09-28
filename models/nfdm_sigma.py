from syslog import LOG_SYSLOG
import torch
import torch.nn as nn
import torch.autograd.functional as AF
from abc import ABC, abstractmethod
from typing import Callable, Tuple
import einops
from typing import Optional
import math 
from .utils import MLPNetwork, SinusoidalPosEmb

# Utilities

class Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, num_layers: int = 5):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SELU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

# --------- sigma schedule helpers ---------

def karras_sigma_grid(num_steps: int, sigma_min: float, sigma_max: float, rho: float, device) -> torch.Tensor:
    """EDM Karras schedule from σ_max → σ_min; returns [N] (no trailing 0)."""
    i = torch.arange(num_steps, device=device)
    sig = (sigma_max**(1/rho) + i/(num_steps-1) * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    return sig  

def rand_log_normal(shape, device, loc=-1.2, scale=1.2, sigma_min=1e-3, sigma_max=80.0):
    """σ ~ LogNormal(loc, scale), clamped to [sigma_min, sigma_max]. Returns [B,1]."""
    z = torch.randn(shape, device=device)
    sigma = torch.exp(loc + scale * z).clamp(min=sigma_min, max=sigma_max).unsqueeze(1)
    return sigma

# Predictor
class Predictor(nn.Module):
    """
    NFDM predictor in σ-space mirroring EDM-style's ScoreNetwork.

    θ̂(z, σ, X) = c_skip(σ) * z + c_out(σ) * Fψ([c_in(σ)*z, embed(c_noise(σ)), X])

    Args:
        x_dim:        dimensionality of θ / z
        hidden_dim:   MLP width
        time_embed_dim: size of sinusoidal embedding vector
        cond_dim:     dimension of conditioning summary s(X)
        cond_mask_prob: classifier-free guidance mask prob (0 disables)
        num_hidden_layers: depth of the MLPNetwork
        sigma_data:   EDM data scale (e.g., 0.5)
        cond_conditional: if False, ignore s and run unconditional
        device:       torch device
    """
    def __init__(self,
                 theta_dim: int,
                 hidden_dim: int,
                 time_embed_dim: int,
                 cond_dim: int,
                 cond_mask_prob: float = 0.0,
                 num_hidden_layers: int = 1,
                 sigma_data: float = 0.5,
                 cond_conditional: bool = True,
                 device: str = "cuda"):
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.cond_mask_prob = cond_mask_prob
        self.cond_conditional = cond_conditional
        self.device = device

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.Mish(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        ).to(device)

        in_dim = theta_dim + time_embed_dim + (cond_dim if cond_conditional else 0)
        self.body = MLPNetwork(
            input_dim=in_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            output_dim=theta_dim,
            device=device,
        ).to(device)

    def _edm_scalings(self, sigma: torch.Tensor):
        sd = self.sigma_data 
        denom = torch.sqrt(sigma ** 2 + sd ** 2) 
        c_skip = (sd ** 2) / (sigma ** 2 + sd ** 2)           
        c_out  = (sigma * sd) / denom                         
        c_in   = 1.0 / denom
        c_noise = 0.25 * torch.log(sigma)
        return c_skip, c_out, c_in, c_noise

    def mask_cond(self, cond, force_mask=False):
        if (not self.training) or (cond is None):
            return cond
        if force_mask:
            return torch.zeros_like(cond)
        if self.cond_mask_prob <= 0:
            return cond
        m = torch.bernoulli(torch.full_like(cond, self.cond_mask_prob))
        return cond * (1. - m)


    def forward(self,
                z: torch.Tensor,          
                sigma: torch.Tensor,      
                s: torch.Tensor = None,   
                uncond: bool = False
                ):

        if sigma.ndim > 1:  
            sigma = sigma.view(-1) 

        # Get EDM scalings 
        c_skip, c_out, c_in, c_noise = self._edm_scalings(sigma.unsqueeze(1))

        emb = self.time_embed(c_noise.view(-1)) #c_noise(σ)          


        if s is not None:
            s = self.mask_cond(s)
            if uncond:
                s = torch.zeros_like(s)

        z_in = c_in * z                           
        if self.cond_conditional and (s is not None):
            h = torch.cat([z_in, s, emb], dim=-1)
        else:
            h = torch.cat([z_in, emb], dim=-1)

        resid_hat = self.body(h)                     

        theta_hat = c_skip * z + c_out * resid_hat   

        return theta_hat

    def get_params(self):
        return self.parameters()

# Volatility Modules

class VolatilityZero(nn.Module):
    """Volatility g(t) ≡ 0."""
    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(sigma)

class VolatilityNeural(nn.Module):
    """
    Learnable volatility g(σ) > 0 with σ-embedding.
    """
    def __init__(self):
        super().__init__()
        self.net = Net(1, 1)
        self.sp = nn.Softplus()

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(1)
        sigma_log = 0.25 * torch.log(sigma)  
        g = self.sp(self.net(sigma_log))
        return g  


class VolatilityNeuralEmbedding(nn.Module):
    """
    Learnable volatility g(σ) > 0 with σ-embedding.
    """
    def __init__(self, embed_dim=16, hidden=64):
        super().__init__()
        self.embed = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2), nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.net = Net(embed_dim, 1, hidden_dim=hidden, num_layers=3)
        self.sp = nn.Softplus()

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(1)
        sigma_log = 0.25 * torch.log(sigma)   
        emb = self.embed(sigma_log)
        g = self.sp(self.net(emb))
        return g  

class VolatilityConst(nn.Module):
    """Constant volatility g(σ) = c > 0."""
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.g.expand_as(sigma)


# Forward Processes

# ----- 1) Affine OT: μ = α(σ) θ, σ_eff = σ  -----

# Note this version does not use JVP to speed up inference.

class AffineOT(nn.Module):
    """OT forward: α(σ) = (σ_max - σ)/(σ_max - σ_min)."""
    def __init__(self, sigma_min=2e-3, sigma_max=80.0):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def alpha(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(1)
        return ((self.sigma_max - sigma) / (self.sigma_max - self.sigma_min))

    def alpha_and_dalpha(self, sigma: torch.Tensor):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(1)
        alpha = ((self.sigma_max - sigma) / (self.sigma_max - self.sigma_min))
        dalpha_dsigma = (-torch.ones_like(sigma)) / (self.sigma_max - self.sigma_min)
        return alpha, dalpha_dsigma

    def forward(self, theta: torch.Tensor, sigma: torch.Tensor, s: Optional[torch.Tensor] = None):
        alpha, _ = self.alpha_and_dalpha(sigma)
        mu = alpha * theta
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(1)
        return mu, sigma #μ = α(σ) θ, σ_eff = σ



# Transform

class TransformOT(nn.Module):
    """PF ops for AffineOT with explicit derivatives (no JVP)."""
    def __init__(self, flow: AffineOT):
        super().__init__()
        self.flow = flow

    def get_t_dir(self, theta: torch.Tensor, sigma: torch.Tensor, s: Optional[torch.Tensor] = None):
        alpha, dalpha_dsigma = self.flow.alpha_and_dalpha(sigma)
        
        mu = alpha * theta 
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(1)
        sig = sigma
        dmu_dsigma = dalpha_dsigma * theta
        dsig_dsigma = torch.ones_like(sig)

        return (mu, sig), (dmu_dsigma, dsig_dsigma)

    def forward(self, eps, sigma, theta, s=None):
        (mu, sig), (dmu, dsig) = self.get_t_dir(theta, sigma, s)

        z = mu + sig * eps 
        dz_dsigma = dmu + dsig * eps
        score = (mu - z) / (sig**2)

        return z, dz_dsigma, score

    def inverse(self, z, sigma, theta, s=None):
        (mu, sig), (dmu, dsig) = self.get_t_dir(theta, sigma, s)

        eps = (z - mu) / (sig) 
        dz_dsigma = dmu + dsig * eps 
        score = (mu - z) / (sig**2)

        return eps, dz_dsigma, score 


# NFDM Core Module

class NeuralDiffusion(nn.Module):
    """
    σ-native NFDM forward pass 
    """
    def __init__(self, transform: nn.Module, pred: nn.Module, vol: nn.Module, stats_collector=None):
        super().__init__()
        self.transform = transform
        self.pred = pred
        self.vol = vol
        self.stats_collector = stats_collector

    def forward(self, theta: torch.Tensor, sigma: torch.Tensor, s: torch.Tensor, 
                collect_stats: bool = False, stats_kwargs: dict = None) -> torch.Tensor:

        B = theta.shape[0]
        eps = torch.randn_like(theta)

        z, f_dz, f_score = self.transform(eps, sigma, theta, s)  

        theta_hat = self.pred(z, sigma, s)

        _, r_dz, r_score = self.transform.inverse(z, sigma, theta_hat, s)

        g = self.vol(sigma)           
        g2 = g ** 2

        f_drift = f_dz - 0.5 * g2 * f_score 
        r_drift = r_dz - 0.5 * g2 * r_score 

        loss = 0.5 * (f_drift - r_drift) ** 2 / g2 #loss = 0.5 * (f_drift - r_drift) ** 2 when using VolatilityZero()
        loss = loss.sum(dim=1) 


        
        # Collect stats
        if collect_stats and self.stats_collector is not None and stats_kwargs is not None:
            self.stats_collector.collect_stats(
                theta=theta,
                sigma=sigma,
                s=s,
                z=z,
                f_dz=f_dz,
                f_score=f_score,
                theta_hat=theta_hat,
                r_dz=r_dz,
                r_score=r_score,
                g=g,
                loss=loss,
                **stats_kwargs
            )
        
        return loss

  