import imageio
import matplotlib.pyplot as plt
import matplotlib.animation
import torch
import numpy as np
from util.tools import batched_autocorrelation


def animate(x, ys, x_lim, y_lim):
    fig = plt.figure()
    plot = plt.plot(x, ys[0].cpu())[0]
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)

    def _inner_plot(frame):
        plot.set_data((x, ys[frame].cpu()))

    return matplotlib.animation.FuncAnimation(fig, _inner_plot, frames=len(ys), interval=25)


def plot_function(x, func):
    fig = plt.figure()
    plt.plot(x, func(x))
    plt.show()
    plt.close()


def viz_fwd_process_diagnostic(fwd_kl_vals, rev_kl_vals, w2_vals, mean_err_vals, cov_matr_err_vals):
    """
    Simple plotting code for visualizing the fwd_process_diagnostic output in tools.py
    """
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    t = torch.arange(1, fwd_kl_vals.shape[0] + 1)

    axs[0, 0].semilogy(t, fwd_kl_vals)
    axs[0, 1].semilogy(t, rev_kl_vals)
    axs[0, 2].semilogy(t, w2_vals)
    axs[1, 0].plot(t, mean_err_vals)
    axs[1, 1].plot(t, cov_matr_err_vals)
    axs[1, 2].axis('off')

    axs[0, 0].set_title(r'$KL[\mathbb{P}_{\infty} || \mathbb{P}_t]$', size='xx-large')
    axs[0, 1].set_title(r'$KL[\mathbb{P}_t || \mathbb{P}_{\infty}]$', size='xx-large')
    axs[0, 2].set_title(r'$W_2(\mathbb{P}_t, \mathbb{P}_\infty)$', size='xx-large')
    axs[1, 0].set_title(r'$||\mu_t - \mu||_2$', size='xx-large')
    axs[1, 1].set_title(r'$||\Sigma_t - \Sigma||_F$', size='xx-large')

    plt.tight_layout()


def viz_samples(x, real_samples, gen_samples):
    fig, axs = plt.subplots(2, 4, figsize=(12, 8), sharey='col')

    n_x = real_samples.shape[1]

    # Plot real samples
    for samp in real_samples:
        axs[0, 0].plot(x, samp)

    # Plot generated samples
    for samp in gen_samples:
        axs[1, 0].plot(x, samp)

    # Plot pointwise means
    axs[0, 1].plot(x, torch.mean(real_samples, axis=0))
    axs[1, 1].plot(x, torch.mean(gen_samples, axis=0))

    # Plot pointwise variances
    axs[0, 2].plot(x, torch.std(real_samples, axis=0))
    axs[1, 2].plot(x, torch.std(gen_samples, axis=0))

    # Plot autocorrelations -- should these be averaged...?
    real_autocorr = torch.as_tensor(batched_autocorrelation(real_samples, normalize=True))
    gen_autocorr = torch.as_tensor(batched_autocorrelation(gen_samples, normalize=True))
    axs[0, 3].plot(np.arange(0, n_x), torch.mean(real_autocorr, axis=0))
    axs[1, 3].plot(np.arange(0, n_x), torch.mean(gen_autocorr, axis=0))

    # Titles, labels, etc.
    axs[0, 0].set_title('Samples', size='x-large')
    axs[0, 1].set_title('Pointwise Mean', size='x-large')
    axs[0, 2].set_title('Pointwise Variance', size='x-large')
    axs[0, 3].set_title('Avg. Autocorrelation', size='x-large')

    axs[0, 0].set_ylabel('Train', size='x-large', rotation=90)
    axs[1, 0].set_ylabel('Generated', size='x-large', rotation=90)

    plt.tight_layout()


def viz_samples_v2(x, real_samples, gen_samples):
    fig, axs = plt.subplots(2, 4, figsize=(12, 8), sharey='col')

    n_x = real_samples.shape[1]

    # Plot real samples
    for samp in real_samples:
        axs[0, 0].plot(x, samp)

    # Plot generated samples
    for samp in gen_samples:
        axs[1, 0].plot(x, samp)

    # Plot pointwise means
    axs[0, 1].plot(x, torch.mean(real_samples, axis=0))
    axs[0, 1].plot(x, torch.mean(gen_samples, axis=0))

    # Plot pointwise variances
    axs[0, 2].plot(x, torch.std(real_samples, axis=0))
    axs[0, 2].plot(x, torch.std(gen_samples, axis=0))

    # Plot autocorrelations
    real_autocorr = torch.as_tensor(batched_autocorrelation(real_samples, normalize=True))
    gen_autocorr = torch.as_tensor(batched_autocorrelation(gen_samples, normalize=True))
    axs[0, 3].plot(np.arange(0, n_x), torch.mean(real_autocorr, axis=0), label='Train')
    axs[0, 3].plot(np.arange(0, n_x), torch.mean(gen_autocorr, axis=0), label='Gen.')

    # Titles, labels, etc.
    axs[0, 0].set_title('Samples', size='x-large')
    axs[0, 1].set_title('Pointwise Mean', size='x-large')
    axs[0, 2].set_title('Pointwise Variance', size='x-large')
    axs[0, 3].set_title('Avg. Autocorrelation', size='x-large')

    axs[0, 0].set_ylabel('Train', size='x-large', rotation=90)
    axs[1, 0].set_ylabel('Generated', size='x-large', rotation=90)

    axs[0, 3].legend()

    axs[1, 1].axis('off')
    axs[1, 2].axis('off')
    axs[1, 3].axis('off')

    plt.tight_layout()
