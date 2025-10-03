from re import X
from syslog import LOG_SYSLOG
import torch
import torch.nn as nn
import torch.autograd.functional as AF
from abc import ABC, abstractmethod
from typing import Callable, Tuple
import einops
from .utils import SinusoidalPosEmb, MLPNetwork
from .normalizing_flow import Conditioner

## MLP ##

class Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, num_layers: int = 5):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SELU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
## FORWARD PROCESS ##

class AffineFlow(nn.Module, ABC):

    @abstractmethod
    def forward(self, theta: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

class AffineOT(AffineFlow):
    def __init__(self, d: int, cond_dim: int, delta: float = 1e-2):
        super().__init__()

    def forward(self, theta: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (1 - t) * theta, (t + (1 - t) * 0.01) * torch.ones_like(theta)
    

class AffineTransform(nn.Module):
    """
    Wraps an AffineFlow to provide (z, dz, score) and inverse (eps, dz, score).
    """
    def __init__(self, flow: AffineFlow):
        super().__init__()
        self.flow = flow

    def get_t_dir(self, theta: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        def f(theta_in):
            def f_(t_in):
                return self.flow(theta_in, t_in, s)
            return f_

        return t_dir(f(theta), t)

    def forward(self, eps: torch.Tensor, t: torch.Tensor, theta: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (mu, sigma), (dmu, dsigma) = self.get_t_dir(theta, t, s)

        z = mu + sigma * eps
        dz = dmu + dsigma * eps
        score = - eps / s

        return z, dz, score

    def inverse(self, z: torch.Tensor, t: torch.Tensor, theta: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (mu, sigma), (dmu, dsigma) = self.get_t_dir(theta, t, s)

        eps = (z - mu) / sigma
        dz = dmu + dsigma / sigma * (z - mu)
        score = (mu - z) / sigma ** 2

        return eps, dz, score
    

class AffineNeural(AffineFlow):

    def __init__(self, d: int, cond_dim: int, delta: float = 1e-2):
        super().__init__()
        self.net = Net(d + 1 + cond_dim, 2 * d)
        self.delta = delta

    def forward(self, theta: torch.Tensor, t: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        xts = torch.cat([theta, t, X], dim=-1)            
        m_ls = self.net(xts)                       
        bar_mu, bar_log_sigma = m_ls.chunk(2, dim=-1) 

        mu = (1 - t) * theta + t * (1 - t) * bar_mu 

        log_sigma = (1 - t) * torch.log(torch.full_like(t, self.delta)) \
                    + t * (1 - t) * bar_log_sigma 
        sigma = torch.exp(log_sigma)
 
        return mu, sigma
    

def _prep_t(t: torch.Tensor) -> torch.Tensor:
    """Ensure t is [B,1]."""
    return t.unsqueeze(1) if t.ndim == 1 else t

# ---------------------------
# Forward flow: Affine OT in t
# ---------------------------

class AffineOTExplicit(nn.Module):
    """
    Affine OT forward in t:
      μ(θ,t)   = (1 - t) * θ
      σ_eff(t) = 0.01 + 0.99 * t
    """
    def __init__(self):
        super().__init__()

    def mu_and_dmu(self, theta: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = _prep_t(t)
        mu  = (1.0 - t) * theta         # [B,D]
        dmu = -theta                    # [B,D]
        return mu, dmu

    def sigma_and_dsigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = _prep_t(t)
        sigma  = 0.01 + 0.99 * t                 # [B,1]
        dsigma = torch.full_like(sigma, 0.99)                # [B,1]
        return sigma, dsigma

    def forward(
        self,
        theta: torch.Tensor,                # [B,D]
        t: torch.Tensor,                    # [B] or [B,1]
        s: torch.Tensor = None    # [B,S] (unused for OT)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, _      = self.mu_and_dmu(theta, t)
        sigma, _   = self.sigma_and_dsigma(t)
        return mu, sigma


class AffineTransformOTExplicit(nn.Module):
    """
    Probability-flow ops for AffineOT in t with explicit derivatives (no JVP):
      forward(eps, t, θ)  -> (z, ∂z/∂t, score_θ)
      inverse(z,  t, θ)   -> (ε, ∂z/∂t|θ̂, score_θ̂)
    """
    def __init__(self, flow: AffineOTExplicit):
        super().__init__()
        self.flow = flow

    def get_t_dir(
        self,
        theta: torch.Tensor,                # [B,D]
        t: torch.Tensor,                    # [B] or [B,1]
        s: torch.Tensor = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        mu, dmu       = self.flow.mu_and_dmu(theta, t)       # [B,D], [B,D]
        sigma, dsigma = self.flow.sigma_and_dsigma(t)         # [B,1], [B,1]
        return (mu, sigma), (dmu, dsigma)

    def forward(
        self,
        eps: torch.Tensor,                 # [B,D]
        t: torch.Tensor,                   # [B] or [B,1]
        theta: torch.Tensor,               # [B,D]
        s: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (mu, sigma), (dmu, dsigma) = self.get_t_dir(theta, t, s)
        z = mu + sigma * eps                                   # [B,D]
        dz = dmu + dsigma * eps                             # [B,D]
        score = -eps / sigma                                   # [B,D]  (Gaussian: (mu - z)/σ^2 = -ε/σ)
        return z, dz, score

    def inverse(
        self,
        z: torch.Tensor,                   # [B,D]
        t: torch.Tensor,                   # [B] or [B,1]
        theta: torch.Tensor,               # [B,D]
        s: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (mu, sigma), (dmu, dsigma) = self.get_t_dir(theta, t, s)
        eps = (z - mu) / sigma                                # [B,D]
        dz = dmu + dsigma / sigma * (z-mu)                    # [B,D]
        score = (mu - z) / (sigma**2)                         # [B,D]
        return eps, dz, score



class VolatilityNeural(nn.Module):
    """Learnable volatility schedule g_phi(t)."""
    def __init__(self):
        super().__init__()
        self.net = Net(1, 1) 
        self.sp = nn.Softplus() 

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.sp(self.net(t))



class Predictor(nn.Module):
    def __init__(
            self,
            theta_dim: int,
            hidden_dim: int,
            time_embed_dim: int,
            cond_dim: int,
            cond_mask_prob: float,
            num_hidden_layers: int = 1,
            output_dim=1,
            device: str = 'cuda',
            cond_conditional: bool = True
    ):
        super(Predictor, self).__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.Mish(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        ).to(device)
        
        self.time_embed_dim = time_embed_dim
        self.cond_mask_prob = cond_mask_prob
        self.cond_conditional = cond_conditional 
        if self.cond_conditional:
            input_dim = self.time_embed_dim + theta_dim + cond_dim
        else:
            input_dim = self.time_embed_dim + theta_dim

        # Use MLPNetwork for better architecture
        self.layers = MLPNetwork(
            input_dim,
            hidden_dim,
            num_hidden_layers,
            theta_dim,  # output_dim = theta_dim (same as original Predictor)
            device
        ).to(device)

        self.device = device
        self.training = True

    def forward(self, z: torch.Tensor, t: torch.Tensor, s: torch.Tensor, uncond: bool = False) -> torch.Tensor:
        # Time embedding (similar to ScoreNetwork)
        if len(t.shape) == 0:
            t = einops.rearrange(t, ' -> 1')  # Use einops like ScoreNetwork
            t = t.unsqueeze(1)
        elif len(t.shape) == 1:
            t = t.unsqueeze(1)
        
        time_embed = self.time_embed(t)
        time_embed.squeeze_(1)  # Added missing squeeze
        if time_embed.shape[0] != z.shape[0]:
            time_embed = einops.repeat(time_embed, '1 d -> (1 b) d', b=z.shape[0])
        
        # Concatenate inputs: [z, time_embed, s] (correct order for Predictor)
        if self.cond_conditional:
            zts = torch.cat([z, s, time_embed], dim=-1)  
        else:
            zts = torch.cat([z, time_embed], dim=-1)
        
        theta_hat = self.layers(zts)

        theta_hat = (1 - t) * z + (t + 0.01) * theta_hat

        return theta_hat

    
class NeuralDiffusion(nn.Module):
    """
    Combines forward transform, predictor, and volatility into NFDM forward pass.
    Returns per-sample drift-matching loss.
    """
    def __init__(self, transform: AffineTransform, pred: Predictor, vol: nn.Module):
        super().__init__()
        self.transform = transform
        self.pred = pred
        self.vol = vol

    def forward(self, theta: torch.Tensor, t: torch.Tensor, s: torch.Tensor, 
                collect_stats: bool = False, stats_kwargs: dict = None) -> torch.Tensor:
        eps = torch.randn_like(theta)

        z, f_dz, f_score = self.transform(eps, t, theta, s) #note s = X in non encoder case, i.e sum of cosines, dirichelet, witches hat.

        theta_hat = self.pred(z, t, s)

        _, r_dz, r_score = self.transform.inverse(z, t, theta_hat, s)

        g = self.vol(t)
        g2 = g ** 2

        f_drift = f_dz - 0.5 * g2 * f_score
        r_drift = r_dz - 0.5 * g2 * r_score

        loss = 0.5 * (f_drift - r_drift) ** 2 / g2 
        loss = loss.sum(dim=1)

        sigma_eff = 0.01 + 0.99 * t  # just to record the effective sigma for stats

        # Collect stats
        if collect_stats and self.stats_collector is not None and stats_kwargs is not None:
            self.stats_collector.collect_stats(
                theta=theta,
                sigma=sigma_eff,
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


# Utilities

def jvp(f: Callable[..., torch.Tensor], x: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.autograd.functional.jvp(
        f, x, v, 
        create_graph=torch.is_grad_enabled()
    )


def t_dir(f: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return jvp(f, t, torch.ones_like(t))

def solve_sde(
    sde: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    z: torch.Tensor,
    ts: float,
    tf: float,
    n_steps: int,
    show_pbar: bool = False,
    use_heun: bool = True
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Expects sde(z, t) -> (f(z,t), g(z,t)).
    """
    device = z.device
    bs = z.shape[0]
    t_steps = torch.linspace(ts, tf, n_steps + 1, device=device)
    dt = (tf - ts) / n_steps
    dt_sqrt = abs(dt) ** 0.5

    path = [z]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        t_c = t_cur.expand(bs, 1)

        f, g = sde(z, t_c)
        w = torch.randn_like(z)
        z_pred = z + f * dt + g * w * dt_sqrt

        if use_heun and i < n_steps - 1: #Heun second order correction

            t_next = t_next.expand(bs, 1)
            f_next, g_next = sde(z_pred, t_next)

            z = z + 0.5 * (f + f_next) * dt + 0.5 * (g + g_next) * w * dt_sqrt
            
        else:

            z = z_pred

        path.append(z)

    return z, (t_steps, torch.stack(path))


def solve_ode(
    ode: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    ts: float,
    tf: float,
    n_steps: int,
    show_pbar: bool = False,
    use_heun: bool = True
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    #Deterministic ODE solver via Euler scheme with optional Heun's method.

    def sde_wrapper(z_in: torch.Tensor, t_in: torch.Tensor):
        return ode(z_in, t_in), torch.zeros_like(z_in)
    return solve_sde(sde_wrapper, z, ts, tf, n_steps, show_pbar, use_heun)
