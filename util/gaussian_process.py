import torch
from torch.distributions import MultivariateNormal


class GaussianProcess:
    def __init__(self, kernel, mean=None, covar_eps=1e-10):
        """
        kernel: a kernel object specifying the covariance structure of the GP
        mean: mean function of GP; defaults to zero
        covar_eps: small scalar added to diagonal of covariance matrix for numerical stability
        """
        self.kernel = kernel
        self.covar_eps = covar_eps

        if mean is not None:
            self.mean = mean
        else:  # Default mean is the zero function
            self.mean = lambda x: torch.zeros_like(x)

    def sample(self, query_points, n_samples=1):
        """ Sample a batch of curves from the GP, evaluated at query_points.
        query_points is assumed to be the same for all of the curves.

        query_points: (n_x, d_x)
        """
        assert len(query_points.shape) == 2, f'Query points should be 2d but got shape {query_points.shape}'

        qp_means = torch.as_tensor(self.mean(query_points)).to(query_points.device)  # (n_x, d_y)
        qp_cov_matrix = self.kernel(query_points, query_points)  # (n_x, n_x)
        qp_cov_matrix = qp_cov_matrix.to(torch.double).to(query_points.device)
        qp_cov_matrix = qp_cov_matrix + self.covar_eps * torch.eye(query_points.shape[0], device=query_points.device)

        # Note: the following is assuming d_y = 1
        qp_means = qp_means.squeeze(-1)
        distr = MultivariateNormal(qp_means, covariance_matrix=qp_cov_matrix)
        samples = distr.sample([n_samples])  # (n_samples, n_x)

        return samples
