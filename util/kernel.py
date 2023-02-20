import torch
import abc
import numpy as np
import scipy.special
import scipy.stats
import warnings

# from util.function import Function
import torch.nn as nn
from math import sqrt
from scipy.spatial.distance import cdist
from scipy.optimize import root_scalar
from sklearn.gaussian_process.kernels import Matern, StationaryKernelMixin
import sklearn.gaussian_process.kernels
from typing import List
from math import cos, sin


class Distribution(abc.ABC):
    """
    Represents a distribution over some Hilbert space with a probability measure.
    """

    @abc.abstractmethod
    def sample(self, size: int) -> np.ndarray:
        """
        Draws a sample of shape (size, ...) where ... is the dimension of each draw. M is the 
        :param size: the number of draws to make from the distribution
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts a tensor of shape (X, D) and returns a (X,) tensor that gives the probability of each element
        """
        raise NotImplementedError


class GaussianDistribution(Distribution):
    def __init__(self, mu, sigma) -> None:
        super().__init__()
        self._sigma = sigma
        self.mu = mu

    def sample(self, size: int):
        return np.random.normal(loc=self.mu, scale=np.sqrt(self._sigma), size=(size, 1))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(scipy.stats.norm.pdf(x, loc=self.mu, scale=self._sigma).flatten())


class UniformDistribution(Distribution):
    def __init__(self, a, b) -> None:
        super().__init__()
        self._a, self._b = a, b

    def sample(self, size: int):
        return np.random.uniform(self._a, self._b, (size, 1))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _x = x.to('cpu')
        return torch.as_tensor(scipy.stats.uniform.pdf(_x, loc=self._a, scale=self._b).flatten()).to(x.device)


class IsotropicGaussian(Distribution):
    def __init__(self, mu, sigmas: list) -> None:
        assert len(mu) == len(
            sigmas), "You need to specify the same number of variances"
        super().__init__()
        self._sigmas = sigmas
        self._mus = np.asarray(mu)
        self._gaussians = [GaussianDistribution(
            m, s) for m, s in zip(mu, sigmas)]

    def sample(self, size):
        return np.concatenate([g.sample(size) for g in self._gaussians], axis=1)


class UniformCompactDistribution(Distribution):
    def __init__(self, a, b) -> None:
        super().__init__()
        self._a, self._b = a, b
        self._distributions = [UniformDistribution(x, y) for x, y in zip(a, b)]

    def sample(self, size: int) -> np.ndarray:
        return np.concatenate([d.sample(size) for d in self._distributions], axis=1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.prod(torch.stack([d(x) for d in self._distributions]), dim=0)


class CompositeDistribution(Distribution):
    def __init__(self, *args: List[Distribution]) -> None:
        super().__init__()
        assert all(isinstance(a, Distribution) for a in args)
        self._distributions = args

    def sample(self, size: int) -> np.ndarray:
        return np.concatenate([d.sample(size) for d in self._distributions], axis=1)


def make_distribution(func):
    """
    Makes a distribution from a sampling algorithm which fits Distribution.sample
    :param func: a lambda that accepts an integer size and gives back a matrix of shape (size, ...) where ... are the 
        element sample dimensions
    """


class Kernel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X, Y) -> torch.Tensor:
        """
        Evaluates the kernel between a 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def eigenfunction(self, k: int):
        """
        Gives the kth eigentuple as (eigenvalue, eigenfunction) pair where k is zero-indexed. 
        :param k: an integer representing the index of the eigentuple. Higher eigenvalues are lower indexed
        """
        raise NotImplementedError


class NystromNumerical:
    def __init__(self, distribution: Distribution, kernel: Kernel, generated: int = 10):
        self._generated = generated
        self._distribution = distribution
        self._kernel = kernel

        self._samples = []
        if generated > 0:
            self._generate(generated)
        else:
            self._kernel = np.zeros(1)
            self._lambda, self._vectors = np.zeros(1), np.zeros(1)

    def _generate(self, n):
        """
        generate a gram matrix of size n x n and find eigenvalues
        """
        while len(self._samples) < n:
            self._samples.append(self._distribution.sample(1).flatten())
        self._numpy_samples = np.asarray(self._samples)
        self._kernel_matrix = self._kernel(
            self._numpy_samples, self._numpy_samples)
        self._lambda, self._vectors = np.linalg.eigh(self._kernel_matrix)
        self._generated = len(self._lambda)

    def __call__(self, k: int) -> float:
        assert k >= 0
        if k >= self._generated:
            self._generate(k + 1)
        assert k < self._generated
        eigval = self._lambda[self._generated - k - 1]

        def phi(x):
            return sqrt(self._generated) / eigval * self._kernel(
                self._numpy_samples, x).T @ self._vectors[:, self._generated - k - 1]

        eigfunc = Function()
        eigfunc.functional_repr = phi

        return (eigval / self._generated, eigfunc)


_root_2 = sqrt(2)


def _se_gaussian_measure_analytic(a, b, c, k: int):
    A = a + b + c
    B = b / A
    hermite = scipy.special.hermite(k)

    def phi(x): return torch.exp(-(c - a) * x ** 2) * hermite(_root_2 * c * x)

    phi_func = Function()
    phi_func.functional_repr = phi
    return (np.sqrt(2 * a / A) * B ** k, phi_func)


def _se_isotropic_gaussian_measure_analytic(a, b, c, k: int):
    """
    """
    # for an isotropic gaussian or one that is at least independent on all axes, we can simply take the product
    # of eigenfunctions and eigenvalues
    eigval = 1
    eigfunc = None
    for x, y, z in zip(a, b, c):
        # intermediate eigenvalue and eigenfunction
        t_eigval, t_eigfunc = _se_gaussian_measure_analytic(x, y, z, k)
        eigval *= t_eigval
        eigfunc = t_eigfunc if eigfunc is None else eigfunc * t_eigfunc
    return (eigval, eigfunc)


class SquaredExponential(Kernel):
    def __init__(self, distribution: Distribution, length, generated=1000, force_numerical=False):
        self._distribution = distribution
        self._length = length

        self._analytic_eigenfunction = None

        if force_numerical:
            self._numerical_eigenfunction = NystromNumerical(
                distribution, self, generated)
            return
        if type(self._distribution) is GaussianDistribution and self._distribution.mu == 0:
            self.__a = 1 / (4 * self._distribution._sigma ** 2)
            self.__b = 1 / (2 * self._length ** 2)
            self.__c = sqrt(self.__a ** 2 + 2 * self.__a * self.__b)
            self._analytic_eigenfunction = lambda k: _se_gaussian_measure_analytic(
                self.__a, self.__b, self.__c, k)
        elif isinstance(self._distribution, IsotropicGaussian) and np.all(self._distribution._mus == 0):
            self.__a == [1 / (4 * s ** 2) for s in self._distribution._sigmas]
            self.__b = [1 / (2 * self._length ** 2)
                        for _ in range(len(self._distribution._sigmas))]
            self.__c = [sqrt(a ** 2 + 2 * a * b)
                        for a, b, in zip(self.__a, self.__b)]
            self._analytic_eigenfunction = lambda k: _se_isotropic_gaussian_measure_analytic(
                self.__a, self.__b, self.__c, k
            )
        else:
            self._numerical_eigenfunction = NystromNumerical(
                distribution, self, generated)

    def __call__(self, x, xprime):
        """
        :param x: a (M, D) matrix where each row represents a vector of the desired dimension
        :param xprime: a (N, D) matrix where each row represents a vector of the desired dimension
        :returns: a (M, N) matrix representing the squared exponential function operated on each pair of vectors from x and xprime
        """
        assert hasattr(x, 'shape'), "You need to provide a tensor type."
        assert len(
            x.shape) == 2, "Two dimensional shapes only (did you reshape for features?)"
        if isinstance(x, torch.Tensor):
            xprime = xprime.clone()
            return torch.exp(-torch.cdist(x.unsqueeze(0), xprime.unsqueeze(0)) ** 2 / (2 * self._length ** 2)).squeeze(
                0)
        return np.exp(-cdist(x, xprime) ** 2 / (2 * self._length ** 2))

    def eigenfunction(self, k: int):
        """
        k is zero-indexed
        """
        if self._analytic_eigenfunction:
            return self._analytic_eigenfunction(k)
        return self._numerical_eigenfunction(k)


class MaternKernel(Kernel):
    def __init__(self, distribution: Distribution, length, generated=50, nu=1.5):
        self._distribution = distribution
        self._length = length
        self._matern = Matern(length=length, nu=nu)
        self._numerical_eigenfunction = NystromNumerical(
            distribution, self, generated)

    def __call__(self, x, xprime):
        """
        :param x: a (M, D) matrix where each row represents a vector of the desired dimension
        :param xprime: a (N, D) matrix where each row represents a vector of the desired dimension
        :returns: a (M, N) matrix representing the squared exponential function operated on each pair of vectors from x and xprime
        """
        return self._matern(x, xprime)

    def eigenfunction(self, k: int):
        """
        k is zero-indexed
        """
        return self._numerical_eigenfunction(k)


class ExponentialKernel(Kernel, StationaryKernelMixin, sklearn.gaussian_process.kernels.Kernel):
    # Implements the exponential kernel w.r.t. the uniform measure on [a, b]
    # see https://www.mlmi.eng.cam.ac.uk/files/burt_thesis.pdf Section 3.5
    def __init__(self, a, b, length, variance, device='cpu'):
        super(ExponentialKernel, self).__init__()
        self._a = a
        self._b = b
        self._distribution = UniformDistribution(a, b)
        self._length = length
        self._device = device
        self._variance = variance

        self.anisotropic = False

    def __call__(self, x, xprime=None, differentiate=False):
        """
        :param x: a (M, D) matrix where each row represents a vector of the desired dimension
        :param xprime: a (N, D) matrix where each row represents a vector of the desired dimension
        :returns: a (M, N) matrix representing the exponential function operated on each pair of vectors from x and xprime

        # Note this is computing k(x, x')
        # Differentiate=True will compute d/dx' k(x, x')
        """
        if xprime is None: xprime = x

        x = x.to(torch.double)
        xprime = xprime.to(torch.double)

        if differentiate:
            # This is the differential covariance matrix
            assert x.shape[1] == 1, 'Differentiation only supported when d_x = 1'
            cov_matr = self(x, xprime=xprime, differentiate=False)
            return cov_matr * torch.sign(x - xprime.T) / self._length
        else:
            if isinstance(x, torch.Tensor):
                return self._variance * torch.exp(
                    -torch.cdist(x.unsqueeze(0), xprime.unsqueeze(0)) / self._length).squeeze(0)
            return self._variance * np.exp(-cdist(x, xprime) / self._length)

    def eigenfunction(self, k: int, order=0):
        """
        k is zero-indexed
        """
        omega = self.compute_omega_root(k)
        eigval = self._variance * 2 * self._length / \
                 (1. + (omega * self._length) ** 2)

        # eigfunc = Function(min_x=self._a, max_x=self._b)

        def _eigenfunction(x, k: int):
            # assume k is one-indexed in here
            c1 = omega * (x - 0.5 * (self._b + self._a))
            c2 = sin(omega * (self._b - self._a)) / omega
            if k % 2 == 0:
                return torch.cos(c1) / sqrt(0.5 * ((self._b - self._a) + c2))
            else:
                return torch.sin(c1) / sqrt(0.5 * ((self._b - self._a) - c2))

        def _eigenfunction_d1(x, k):
            # First derivative of eigenfunction
            # assume k is one-indexed in here
            c1 = omega * (x - 0.5 * (self._b + self._a))
            c2 = sin(omega * (self._b - self._a)) / omega
            if k % 2 == 0:
                return -1. * omega * torch.sin(c1) / sqrt(0.5 * ((self._b - self._a) + c2))
            else:
                return omega * torch.cos(c1) / sqrt(0.5 * ((self._b - self._a) - c2))

        def _eigenfunction_d2(x, k):
            # Second derivative of eigenfunction
            c1 = omega * (x - 0.5 * (self._b + self._a))
            c2 = sin(omega * (self._b - self._a)) / omega
            if k % 2 == 0:
                return -1. * omega ** 2 * torch.cos(c1) / sqrt(0.5 * ((self._b - self._a) + c2))
            else:
                return -1. * omega ** 2 * torch.sin(c1) / sqrt(0.5 * ((self._b - self._a) - c2))

        eigfunc = lambda x: _eigenfunction(x, k)
        eigfunc_d1 = lambda x: _eigenfunction_d1(x, k)
        eigfunc_d2 = lambda x: _eigenfunction_d2(x, k)
        # eigfunc.functional_repr = lambda x: _eigenfunction(x, k)

        if order == 0:
            return eigval, eigfunc
        elif order == 1:
            return eigval, eigfunc, eigfunc_d1
        elif order == 2:
            return eigval, eigfunc, eigfunc_d1, eigfunc_d2
        else:
            raise NotImplementedError(f'Order {order} not supported')

    def compute_omega_root(self, k: int):
        """
        assume k zero indexed
        """
        eps = 1e-6

        if k % 2 == 1:
            bracket_lb = k * np.pi / (self._b - self._a) + eps
            bracket_ub = (k + 2) * np.pi / (self._b - self._a) - eps
        else:
            if k == 0:
                bracket_lb = 0 + eps
            else:
                bracket_lb = (k - 1) * np.pi / (self._b - self._a) + eps
            bracket_ub = (k + 1) * np.pi / (self._b - self._a) - eps

        root_find_res = root_scalar(self.omega_fxn, bracket=[
            bracket_lb, bracket_ub], args=(k,))

        # Some safety checks
        if not root_find_res.converged:
            warnings.warn(
                'Root finding failed to converge in Exponential Kernel.')

        return root_find_res.root

    def omega_fxn(self, omega, k: int):
        # Eqns. 3.16-3.17 in https://www.mlmi.eng.cam.ac.uk/files/burt_thesis.pdf
        # Assumes k is zero-indexed

        if k % 2 == 0:
            return self._length * omega * np.tan(0.5 * omega * (self._b - self._a)) - 1.
        else:
            return self._length * omega + np.tan(0.5 * omega * (self._b - self._a))

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self._length)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self._length)[0]
            )

    def diag(self, X):
        length_scale = self._length
        X = torch.as_tensor(X)
        dists = torch.cdist(X.unsqueeze(1), X.unsqueeze(1)).squeeze(1)
        K = self._variance ** 2 * torch.exp(-dists / length_scale)
        if isinstance(X, np.ndarray):
            return np.asarray(K)
        return K

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return True


class Matern32(Kernel, StationaryKernelMixin, sklearn.gaussian_process.kernels.Kernel):
    # Implements the matern kernel with nu = 3/2.
    # See R&W, Chapter 4.2
    def __init__(self, a, b, length, variance, device='cpu'):
        super(Matern32, self).__init__()
        self._a = a
        self._b = b
        self._distribution = UniformDistribution(a, b)
        self._length = length
        self._device = device
        self._variance = variance

        self.anisotropic = False

    def __call__(self, x, xprime=None, differentiate=False):
        """
        :param x: a (M, D) matrix where each row represents a vector of the desired dimension
        :param xprime: a (N, D) matrix where each row represents a vector of the desired dimension
        :returns: a (M, N) matrix representing the exponential function operated on each pair of vectors from x and xprime

        # Note this is computing k(x, x')
        # Differentiate=True will compute d/dx' k(x, x')
        """
        if xprime is None: xprime = x

        x = x.to(torch.double)
        xprime = xprime.to(torch.double)

        if differentiate:
            # This is the differential covariance matrix -- differentiated along the argument x'
            assert x.shape[1] == 1, 'Differentiation only supported when d_x = 1'
            r = torch.cdist(x, xprime)
            x_diff = x - xprime.T
            return 3 * (self._variance / self._length ** 2) * x_diff * torch.exp(-np.sqrt(3) * r / self._length)
        else:
            r = torch.cdist(x, xprime)
            c = (1. + np.sqrt(3) * r / self._length)
            return self._variance * c * torch.exp(-np.sqrt(3) * r / self._length)

    def eigenfunction(self, k: int, order=0):
        raise NotImplementedError

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self._length)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self._length)[0]
            )

    def diag(self, X):
        raise NotImplementedError

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return True


class RBF(Kernel, StationaryKernelMixin, sklearn.gaussian_process.kernels.Kernel):
    # Implements the matern kernel with nu = 3/2.
    # See R&W, Chapter 4.2
    def __init__(self, a, b, length, variance, device='cpu'):
        super(RBF, self).__init__()
        self._a = a
        self._b = b
        self._distribution = UniformDistribution(a, b)
        self._length = length
        self._device = device
        self._variance = variance

        self.anisotropic = False

    def __call__(self, x, xprime=None, differentiate=False):
        """
        :param x: a (M, D) matrix where each row represents a vector of the desired dimension
        :param xprime: a (N, D) matrix where each row represents a vector of the desired dimension
        :returns: a (M, N) matrix representing the exponential function operated on each pair of vectors from x and xprime

        # Note this is computing k(x, x')
        # Differentiate=True will compute d/dx' k(x, x')
        """
        if xprime is None: xprime = x

        x = x.to(torch.double)
        xprime = xprime.to(torch.double)

        if differentiate:
            # This is the differential covariance matrix -- differentiated along the argument x'
            assert x.shape[1] == 1, 'Differentiation only supported when d_x = 1'
            x_diff = x - xprime.T
            return x_diff * self(x, xprime, differentiate=False) / self._length**2
        else:
            r = torch.cdist(x, xprime)
            return self._variance * torch.exp(- r**2 / (2 * self._length**2))

    def eigenfunction(self, k: int, order=0):
        raise NotImplementedError

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self._length)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self._length)[0]
            )

    def diag(self, X):
        raise NotImplementedError

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return True