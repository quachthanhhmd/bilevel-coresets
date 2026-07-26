"""Nystrom proxy feature maps (Sec. 3.5.2).

The coreset selection of Eq. (10) is solved on a proxy: the RKHS of a kernel
``kappa`` approximated by a ``q``-dimensional Nystrom feature map

    z(.) = D^{-1/2} U^T [kappa(., x_i), i in Q],   K_{Q,Q} = U D U^T,

with the basis ``Q`` drawn uniformly at random ("we use the simplest and
computationally most efficient method of uniform sampling for selecting Q").
On this feature space the inner problem is (strongly) convex multiclass
logistic regression, so it can be solved to a certifiable tolerance.

The paper uses the CNTK of Arora et al. (2019) as ``kappa``; that requires
``jax``/``neural-tangents`` (see ``cl_streaming/ntk_generator.py``).  For
lightweight runs an RBF kernel is provided instead -- Table 10 of the paper
shows RBF is a good proxy on MNIST-like data, though it fails on CIFAR-10.
"""

import numpy as np


def make_rbf_kernel(gamma=1e-3):
    """``kappa(x, y) = exp(-gamma ||x - y||^2)`` on flattened inputs."""

    def kernel_fn(X, Y):
        X = np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)
        Y = np.asarray(Y, dtype=np.float64).reshape(Y.shape[0], -1)
        sq = (X ** 2).sum(axis=1)[:, None] + (Y ** 2).sum(axis=1)[None, :] - 2.0 * X @ Y.T
        return np.exp(-gamma * np.maximum(sq, 0.0))

    return kernel_fn


def make_linear_kernel():
    """``kappa(x, y) = <x, y>`` on flattened inputs."""

    def kernel_fn(X, Y):
        X = np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)
        Y = np.asarray(Y, dtype=np.float64).reshape(Y.shape[0], -1)
        return X @ Y.T

    return kernel_fn


class NystromFeatureMap(object):
    """Nystrom feature map for a given kernel and basis.

    Args:
        kernel_fn (callable): ``kernel_fn(X, Y) -> (len(X), len(Y))`` Gram matrix.
        basis (np.ndarray): the ``q`` basis points ``Q``.
        eps (float): floor on the eigenvalues of ``K_{Q,Q}``.
    """

    def __init__(self, kernel_fn, basis, eps=1e-7):
        self.kernel_fn = kernel_fn
        self.basis = basis
        k = kernel_fn(basis, basis)
        u, s, v = np.linalg.svd(k)
        s = np.maximum(s, eps)
        self.normalization = np.dot(u / np.sqrt(s), v)

    def __call__(self, X):
        k = self.kernel_fn(np.asarray(X), self.basis)
        return np.dot(k, self.normalization.T).astype(np.float32)

    @property
    def dim(self):
        return self.normalization.shape[0]


def nystrom_feature_map(kernel_fn, X, q, rs=None, eps=1e-7):
    """Build a Nystrom feature map from ``q`` uniformly sampled points of ``X``."""
    rs = np.random if rs is None else rs
    n = len(X)
    q = min(q, n)
    inds = rs.choice(n, q, replace=False)
    basis = np.asarray(X)[inds]
    return NystromFeatureMap(kernel_fn, basis, eps=eps)
