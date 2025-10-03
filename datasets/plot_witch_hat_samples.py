import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize


def plot_witch_hat_samples(samples, model_name, bounding_box=(0.1, 0.9), zoom_out=False, save_path=None):
    """
    Plot Witch's Hat samples for a given model.

    Args:
        samples (np.ndarray): Array of shape (N, 2) with θ₁, θ₂ samples.
        model_name (str): Name of the model (e.g., 'cNF', 'cDiff', 'NFDM').
        bounding_box (tuple): (low, high) bounds for the prior support.
        zoom_out (bool): Whether to include a zoomed-out inset.
        save_path (str or None): If provided, save the figure to this path.
    """
    assert samples.shape[1] == 2, "Samples must be 2D (θ₁, θ₂)."
    low, high = bounding_box

    fig, ax = plt.subplots(figsize=(6, 6))

    # Main hexbin plot
    hb = ax.hexbin(samples[:, 0], samples[:, 1], gridsize=50, cmap='RdYlBu', linewidths=0.5, mincnt=1)
    cb = fig.colorbar(hb, ax=ax, orientation='vertical')
    cb.set_label('Density', fontsize=20, rotation=270, labelpad=30)
    cb.ax.tick_params(labelsize=16)
    cb.set_ticks([])
    cb.ax.text(2.5, 1.01, 'High', ha='center', va='bottom', fontsize=18, color='#B40426', weight='bold', transform=cb.ax.transAxes)
    cb.ax.text(2.5, -0.01, 'Low', ha='center', va='top', fontsize=18, color='#3B4CC0', weight='bold', transform=cb.ax.transAxes)

    # Bounding box for prior support
    rect = Rectangle((low, low), high - low, high - low, linewidth=2, edgecolor='gold', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # Axis labels
    ax.set_xlabel(r'$\theta_1$', fontsize=20)
    ax.set_ylabel(r'$\theta_2$', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

    # Model annotation
    ax.set_title(f"{model_name} sampling", fontsize=16)

    # Optional zoomed-out inset
    if zoom_out:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="35%", height="35%", loc='lower left', borderpad=2)
        axins.hexbin(samples[:, 0], samples[:, 1], gridsize=30, cmap='RdYlBu_r', linewidths=0.2, mincnt=1)
        axins.add_patch(Rectangle((low, low), high - low, high - low, linewidth=1.5, edgecolor='gold', facecolor='none', linestyle='--'))
        axins.set_xlim(samples[:, 0].min() - 0.1, samples[:, 0].max() + 0.1)
        axins.set_ylim(samples[:, 1].min() - 0.1, samples[:, 1].max() + 0.1)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title('zoomed out', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()