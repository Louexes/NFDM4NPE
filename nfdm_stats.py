import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import os


class NFDMStatsCollector:
    """
    Centralized stats collector for NFDM metrics during training.
    Accumulates metrics from forward passes and provides formatted output.
    """
    
    def __init__(self, save_path: str, use_wandb: bool = False, wandb_run=None):
        self.save_path = save_path
        self.use_wandb = use_wandb
        self.wandb_run = wandb_run
        self.current_stats = {}
        self.stats_history = []
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
    
    def collect_stats(self, 
                     theta: torch.Tensor,
                     sigma: torch.Tensor,
                     s: torch.Tensor,
                     z: torch.Tensor,
                     f_dz: torch.Tensor,
                     f_score: torch.Tensor,
                     theta_hat: torch.Tensor,
                     r_dz: torch.Tensor,
                     r_score: torch.Tensor,
                     g: torch.Tensor,
                     loss: torch.Tensor,
                     epoch: int,
                     seed: int,
                     learning_rate: float,
                     grad_norm: float = 0.0):
        """
        Collect stats from a forward pass. Called during training.
        
        Args:
            theta: Ground truth parameters [B, D]
            sigma: Noise levels [B, 1] 
            s: Conditioning summary [B, S]
            z: Latent variables [B, D]
            f_dz: Forward path derivative [B, D]
            f_score: Forward score [B, D]
            theta_hat: Predicted parameters [B, D]
            r_dz: Reverse path derivative [B, D]
            r_score: Reverse score [B, D]
            g: Volatility [B, 1]
            loss: Per-sample loss [B]
            epoch: Current epoch
            seed: Random seed
            learning_rate: Current learning rate
            grad_norm: Gradient norm (set by trainer)
        """
        with torch.no_grad():
            # Compute all metrics
            g2 = g ** 2
            f_drift = f_dz - 0.5 * g2 * f_score
            r_drift = r_dz - 0.5 * g2 * r_score
            drift_diff = f_drift - r_drift
            drift_diff_norm = drift_diff.norm(dim=1)
            
            # Store current stats
            self.current_stats = {
                # Training metadata
                'seed': int(seed),
                'epoch': int(epoch),
                'learning_rate': round(learning_rate, 3),
                'grad_norm': round(grad_norm, 3),
                'loss': round(loss.mean().item(), 3),
                'theta_hat_mse': round(((theta_hat - theta) ** 2).mean().item(), 3),

                # Sigma metrics (noise level)
                'sigma_mean': round(sigma.mean().item(), 3),
                'sigma_std': round(sigma.std().item(), 3), 
                'sigma_min': round(sigma.min().item(), 3),
                'sigma_max': round(sigma.max().item(), 3),
                
                # Core metrics (g = volatility)
                'g^2_mean': round(g2.mean().item(), 3),
                'g^2_std': round(g2.std().item(), 3),
                'g^2_min': round(g2.min().item(), 3),
                'g^2_max': round(g2.max().item(), 3),
                
                # Drift metrics
                'f_drift_mean': round(f_drift.mean().item(), 3),
                'f_drift_std': round(f_drift.std().item(), 3),
                'f_drift_min': round(f_drift.min().item(), 3),
                'f_drift_max': round(f_drift.max().item(), 3),
                'r_drift_mean': round(r_drift.mean().item(), 3),
                'r_drift_std': round(r_drift.std().item(), 3),
                'r_drift_min': round(r_drift.min().item(), 3),
                'r_drift_max': round(r_drift.max().item(), 3),
                
                # Norm metrics
                'drift_diff_norm_mean': round(drift_diff_norm.mean().item(), 3),
                'drift_diff_norm_std': round(drift_diff_norm.std().item(), 3),
                'drift_diff_norm_min': round(drift_diff_norm.min().item(), 3),
                'drift_diff_norm_max': round(drift_diff_norm.max().item(), 3),
                'score_norm_mean': round(f_score.norm(dim=1).mean().item(), 3),
                
            }
            
            # Add to history
            self.stats_history.append(self.current_stats.copy())
    
    def log_to_wandb(self, epoch: int):
        """Log current stats to wandb."""
        if self.use_wandb and self.wandb_run and self.current_stats:
            wandb_log_data = {'epoch': epoch}
            for key, value in self.current_stats.items():
                if key != 'epoch':  # Don't duplicate epoch
                    wandb_log_data[f'nfdm/{key}'] = value
            self.wandb_run.log(wandb_log_data)
    
    def save_to_csv(self):
        """Save stats history to CSV file."""
        if self.stats_history:
            df = pd.DataFrame(self.stats_history)
            csv_path = os.path.join(self.save_path, 'nfdm_stats.csv')
            df.to_csv(csv_path, index=False)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get the most recent stats."""
        return self.current_stats.copy()
    
    def clear_history(self):
        """Clear the stats history (useful for memory management)."""
        self.stats_history.clear()