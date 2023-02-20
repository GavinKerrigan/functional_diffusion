"""
Contains a few methods for generating synthetic datasets.
"""

import numpy as np
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import shuffle


def sample_mixture_gps(n_samples=300, n_x=100, grid_x=False):
    """ Samples a synthetic dataset consisting of a mixture of two GP distributions.

    Note: n_samples is the number of samples in each component of the mixture, /not/ the total number of samples.
    """
    alpha = 1e-6  # Added to diag. of covariance matr.

    var1 = 0.4
    var2 = 0.4
    length_scale1 = 0.1
    length_scale2 = 0.1
    def mean1(x): return 10*x - 5
    def mean2(x): return -10*x + 5

    kernel1 = var1 * RBF(length_scale=length_scale1)
    kernel2 = var2 * RBF(length_scale=length_scale2)
    gp1 = GaussianProcessRegressor(kernel=kernel1, alpha=alpha)
    gp2 = GaussianProcessRegressor(kernel=kernel2, alpha=alpha)

    rng = np.random.default_rng()
    if grid_x:
        x = np.linspace(0., 1., n_x)
        x = np.tile(x, (n_samples, 1))
        x1 = x2 = x
    else:
        # Generate random observation points
        # The observation points differ for each sampled function

        x1 = rng.uniform(size=(n_samples, n_x))
        x2 = rng.uniform(size=(n_samples, n_x))
        # Sorting is not essential, but makes life easier for e.g. plotting
        x1.sort(axis=1)
        x2.sort(axis=1)

    # Draw samples from GP Priors
    samples_1 = np.empty((n_samples, n_x))
    samples_2 = np.empty((n_samples, n_x))
    for i in range(n_samples):
        samples_1[i, :] = gp1.sample_y(
            x1[i, :].reshape(-1, 1), random_state=None).squeeze(-1) + mean1(x1[i, :])
        samples_2[i, :] = gp2.sample_y(
            x2[i, :].reshape(-1, 1), random_state=None).squeeze(-1) + mean2(x2[i, :])

    x = np.vstack((x1, x2))  # (2 * n_samples, n_x)
    y = np.vstack((samples_1, samples_2))  # (2 * n_samples, n_x)
    x, y = shuffle(x, y)
    return torch.as_tensor(x), torch.as_tensor(y)


def sample_linear_fxns(n_samples=1, n_x=100, grid_x=False):
    """ Samples a synthetic dataset consiting of linear functions with random slopes and biases.
    """
    mean_slope = 2.
    sigma_slope = 0.25
    mean_bias = -1.
    sigma_bias = 0.07

    if grid_x:
        x = torch.linspace(0., 1., n_x)
        x = torch.tile(x, (n_samples, 1))
    else:
        x = torch.rand((n_samples, n_x))  # Uniformly distributed x on [0, 1)

    gaussian_slope = torch.distributions.Normal(mean_slope, sigma_slope)
    gaussian_bias = torch.distributions.Normal(mean_bias, sigma_bias)

    samples_slope = gaussian_slope.sample((n_samples,))
    samples_bias = gaussian_bias.sample((n_samples,))

    y = torch.einsum('ij,i->ij', x, samples_slope) + samples_bias.unsqueeze(1)
    return x, y


def sample_zero_gps(n_samples=300, n_x=256):
    kernel = 0 * RBF(length_scale=1) * 0.001
    gp1 = GaussianProcessRegressor(kernel=kernel)

    x1 = np.random.uniform(size=(n_samples, n_x))
    x1.sort(axis=1)

    # Draw samples from GP Priors
    samples_1 = np.empty((n_samples, n_x))
    for i in range(n_samples):
        samples_1[i, :] = 0 * gp1.sample_y(
            x1[i, :].reshape(-1, 1), random_state=None).squeeze(-1)

    return x1, samples_1
