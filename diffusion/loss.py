import torch
from util.kernel import Distribution, Kernel
from util.tools import make_ddx_operator


def gp_kl(mu1, mu2, dist: Distribution, kernel: Kernel, n_terms=50) -> torch.Tensor:
    # Computes the KL[ GP(mu1, kernel)  ||  GP(mu2, kernel) ] via the Feldman-Hajek theorem

    kl = 0.
    for k in range(n_terms):
        eigenval, eigenfunc = kernel.eigenfunction(k)
        mean_diff_eigenbasis = l2_inner_product(
            mu1 - mu2, eigenfunc, dist) ** 2  # <mu1 - mu2, phi_k> = (m_1k - m_2k)
        kl += mean_diff_eigenbasis / eigenval

    return 0.5 * kl


def l2_inner_product(fxn1, fxn2, dist: Distribution, n_x=1000, device='cpu'):
    # Computes the L^2 inner product \int mu_1(x) mu_2(x) dx

    # x = torch.linspace(fxn1.min_x, fxn1.max_x, n_x).to(device)
    x = torch.linspace(0, 1, n_x).to(device)
    prod_fxn_vals = fxn1(x).to(x.device) * \
                    fxn2(x).to(x.device) * dist(x).to(x.device)
    inner_prod = torch.trapezoid(prod_fxn_vals, x=x)
    return inner_prod


def sobolev_inner_product(fxn1_list, fxn2_list, n_x=1000, device='cpu'):
    # Computes the L^2 inner product \int mu_1(x) mu_2(x) dx
    # assumes fx1_list, fxn2_list are lists of functions and their derivatives

    x = torch.linspace(0, 1, n_x).to(device)
    out = 0.0
    for f1, f2 in zip(fxn1_list, fxn2_list):
        prod_fxn_vals = f1(x).to(x.device) * f2(x).to(x.device)
        l2_inner_prod = torch.trapezoid(prod_fxn_vals, x=x)
        out += l2_inner_prod

    return out


class SpectralLoss:
    def __init__(self, kernel: Kernel, distribution: Distribution, n_terms: int = 300,
                 support: torch.Tensor = torch.linspace(0, 1, 256),
                 dtype: torch.dtype = torch.double,
                 device='cpu') -> None:
        eigenvalues = []
        eigenvectors = []
        for k in range(n_terms):
            e, v = kernel.eigenfunction(k)
            eigenvalues.append(e)
            eigenvectors.append(v)

        # it is more efficient to precompute these particular values
        self.__support = support.reshape((-1, 1))

        # distribution is shape (X,)
        self.__distribution = distribution(self.__support).flatten().to(device)

        # eigenvalues is shape (D,) since we want to divide each eigenvalue contribution in the end
        self.__eigenvalues = torch.tensor(
            eigenvalues, dtype=dtype).flatten().to(device)
        self.__eigenvectors = torch.cat(
            [f(self.__support).reshape((1, -1)) for f in eigenvectors], dim=0).to(device)

        self.__eg_dist = self.__eigenvectors * self.__distribution

        # we actually need the flattened version of support
        self.__support = self.__support.flatten().to(device)

    def __call__(self, mu1: torch.Tensor, mu2: torch.Tensor, debug=False):
        """
        Assume that mu1 and mu2 are evaluations in the shape (B, X) where X is the same shape as support
        """
        assert mu1.shape[0] == mu2.shape[0], f"batch sizes differ: {mu1.shape[0]}, {mu2.shape[0]}"
        assert mu1.shape[1] == len(self.__support), f'mu1 shape {mu1.shape}, support {len(self.__support)}'
        assert len(mu1.shape) == 2

        # (mu1 - mu2) is shape (B, X)
        # eigenvectors are shape (D, X) where D is the n_terms
        # so we unsqueeze the middle dimension to allow it to distribute giving
        prod_fxn_vals = (mu1 - mu2).unsqueeze(1) * self.__eg_dist
        integral = torch.trapezoid(prod_fxn_vals, self.__support)
        return 0.5 * torch.sum(integral ** 2 / self.__eigenvalues, dim=-1)


class DiscreteLoss:
    """ Implements the finite dimensional approximation to GPKL.
    Assumes all function observations occur on the same support.
    """

    def __init__(self, kernel, support, device='cpu'):
        self.support = support  # (n_x, 1)
        self.device = device

        # todo may need double
        covar_matr = kernel(support, support)
        self.cov_matr_inv = torch.linalg.inv(covar_matr).to(self.device)  # (n_x, n_x)

    def __call__(self, mu1, mu2):
        """
        mu1, mu2: (batch_size, n_x)
        """
        assert mu1.shape[0] == mu2.shape[0], f"batch sizes differ: {mu1.shape[0]}, {mu2.shape[0]}"
        assert mu1.shape[1] == mu2.shape[1] == len(
            self.support), f'Shapes {mu1.shape, mu2.shape} do not match support {self.support.shape}'

        diff = (mu1 - mu2).double()
        # Computes quadratic form (mu1 - mu2)^T Cov^{-1} (mu_1 - mu_2) in batched fashion
        loss = torch.einsum('bj,ij,bi->b', diff, self.cov_matr_inv, diff)
        return 0.5 * loss
    

class DiscreteSobolevLoss:
    """ Implements the discrete approximation to the Sobolev-KL loss.
    Note this is only for order one.
    Assumes all function observations occur on the same support.
    """

    def __init__(self, kernel, support, device='cpu'):
        assert support.shape[1] == 1, 'Only supported for d_x = 1'
        self.support = support  # (n_x, 1)
        self.device = device

        self.cov_matr = kernel(support, support).to(self.device).double()
        self.cov_matr_ddx = kernel(support, support, differentiate=True).double().to(self.device)
        dx = support[1:] - support[:-1]
        self.ddx_matrix = make_ddx_operator(support.shape[0], dx=dx).to(self.device).double()
        self.discrete_op = self.cov_matr + torch.matmul(self.cov_matr_ddx, self.ddx_matrix)
        self.inv_op = torch.linalg.inv(self.discrete_op).to(self.device)  # (n_x, n_x)
        self.sobolev_form = self.ddx_matrix.T @ self.ddx_matrix @ self.inv_op  # D^T D C^{-1}
        self.quadratic_form = self.inv_op + self.sobolev_form

        # The quadratic form is not symmetric, PSD in general. So let's project it.
        # https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/
        # https://scicomp.stackexchange.com/questions/30631/how-to-find-the-nearest-a-near-positive-definite-from-a-given-matrix
        self.symmetrized_quadratic_form = 0.5 * (self.quadratic_form + self.quadratic_form.T)
        L, V = torch.linalg.eig(self.symmetrized_quadratic_form)
        assert torch.all(torch.isreal(V)), 'Got complex eigenvectors'
        assert torch.all(torch.isreal(L)), 'Got complex eigenvalues'
        V = V.double()
        L = L.double()
        eps = 1e-5
        L[L < eps] = eps
        self.quadratic_form_proj = V @ torch.diag_embed(L) @ V.T

    def __call__(self, mu1, mu2):
        """
        mu1, mu2: (batch_size, n_x)
        """
        assert mu1.shape[0] == mu2.shape[0], f"batch sizes differ: {mu1.shape[0]}, {mu2.shape[0]}"
        assert mu1.shape[1] == mu2.shape[1] == len(
            self.support), f'Shapes {mu1.shape, mu2.shape} do not match support {self.support.shape}'

        diff = (mu1 - mu2).double()

        # Computes quadratic form (mu1 - mu2)^T Cov^{-1} (mu_1 - mu_2) in batched fashion
        loss = torch.einsum('bj,ij,bi->b', diff, self.quadratic_form_proj, diff)
        
        return 0.5 * loss
