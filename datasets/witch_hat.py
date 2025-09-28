import numpy as np
import sys
sys.path.append("../../")
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader

# Dictionary of hyperparameters for the Witch's Hat distribution
hyperpar_dict = {
    "d": 2,  # Dimension
    "sigma": 0.02,  # Standard deviation of the Gaussian
    "delta": 0.05  # Weight of the uniform component
}

def sample_witch_hat(theta, n_samples, d, sigma, delta):
    # Determine the number of samples to draw from each component using a binomial distribution
    uniform_samples_count = np.random.binomial(n_samples, delta)
    gaussian_samples_count = n_samples - uniform_samples_count

    # Sample from the uniform distribution
    uniform_samples = np.random.uniform(0, 1, (uniform_samples_count, d))

    # Sample from the multivariate Gaussian distribution
    gaussian_samples = np.random.multivariate_normal(theta, sigma ** 2 * np.eye(d), gaussian_samples_count)

    # Combine the samples
    samples = np.vstack((uniform_samples, gaussian_samples))

    # Shuffle the samples to ensure random mixing
    np.random.shuffle(samples)

    return samples

def sample_witch_hat_parameters(input_dict):
    """
    Sample parameters for the Witch's Hat distribution.

    Parameters:
    input_dict (dict): Dictionary containing the hyperparameters for the distributions.

    Returns:
    dict: Dictionary containing the sampled parameters.
    """

    # Extract hyperparameters
    d = input_dict["d"]
    sigma = input_dict["sigma"]
    delta = input_dict["delta"]

    # Sample the location parameter theta uniformly on the unit hypercube [0.1, 0.9]^d
    theta = np.random.uniform(0.1, 0.9, d)

    # Return the sampled parameters in a dictionary
    return {"theta": theta}



def my_gen_sample_size(n, low=100, high=1000):
    return np.random.randint(low=low, high=high, size=n)


def sample_theta():
    return sample_witch_hat_parameters(hyperpar_dict)


def sample_witch_hat_data(parms_dict, sample_size):
    """
    Generate samples from the Witch's Hat distribution using the sampled parameters.

    Parameters:
    parms_dict (dict): Dictionary containing the sampled parameters.
    sample_size (int): The number of samples to generate.

    Returns:
    np.array: Array of generated samples.
    """

    # Extract parameters
    theta = parms_dict["theta"]

    # Globally set hyperparameters
    d = hyperpar_dict["d"]
    sigma = hyperpar_dict["sigma"]
    delta = hyperpar_dict["delta"]

    # Determine the number of samples to draw from each component using a binomial distribution
    uniform_samples_count = np.random.binomial(sample_size, delta)
    gaussian_samples_count = sample_size - uniform_samples_count

    # Sample from the uniform distribution
    uniform_samples = np.random.uniform(0, 1, (uniform_samples_count, d))

    # Sample from the multivariate Gaussian distribution
    gaussian_samples = np.random.multivariate_normal(theta, sigma ** 2 * np.eye(d), gaussian_samples_count)

    # Combine the samples
    samples = np.vstack((uniform_samples, gaussian_samples))

    # Shuffle the samples to ensure random mixing
    np.random.shuffle(samples)

    return samples

def return_witch_hat_dl(n_batches = 256,batch_size = 128,n_sample=None,return_ds=False):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=100, high=1000):
            return np.random.randint(low=low, high=high, size=n)
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_witch_hat_data, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl,ds
    else:
        return dl

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


if __name__ == "__main__":
    dl = return_witch_hat_dl(n_batches=2,batch_size=2,n_sample=1)
    for batch in dl:
        theta, y = batch
        print(theta.shape)
        print(y.shape)
