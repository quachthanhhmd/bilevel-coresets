"""Nystrom approximation of a kernel feature map (Sec. 3.5.2).

Turns a kernel function ``k(x, y)`` (e.g. the CNTK of :mod:`bicoreset.cntk`)
into an explicit ``q``-dimensional feature map ``phi`` such that
``phi(x) . phi(y) ~= k(x, y)``. This lets a kernel method (like the CNTK-based
coreset construction of Sec. 5.1) be re-expressed as an ordinary linear model
on top of ``phi(x)`` -- e.g. multiclass logistic regression, ``theta in
R^{q x c}`` -- exactly the setup Sec. 5.1 uses for CIFAR-10.

This follows the same construction already used elsewhere in this repository
for the NTK-proxy coreset method (``bilevel_coreset.BilevelCoreset
.select_nystrom_batch`` / ``.map_to_nystrom_features``), refactored here into a
standalone utility with no torch-training-loop coupling: given ``q`` landmark
points ``L``,

    phi(x) = k(x, L) @ K_LL^{-1/2}

with ``K_LL^{-1/2}`` obtained from the SVD of the (symmetric, PSD in theory)
landmark Gram matrix ``K_LL``, clipping singular values to a small floor for
numerical stability (kernels computed with finite-precision arithmetic can have
tiny negative or near-zero singular values even though the true kernel is PSD).
"""

import numpy as np


class NystromFeatureMap(object):
    """Explicit feature map ``phi`` approximating a kernel ``k`` via Nystrom.

    Args:
        landmarks: the ``q`` points ``L`` (in whatever format ``kernel_fn``
            accepts, e.g. a ``(q, H, W, C)`` array of images).
        kernel_fn (callable): ``kernel_fn(X, Y) -> (len(X), len(Y))`` kernel
            matrix, e.g. :func:`bicoreset.cntk.cntk6_gap_kernel_fn`.
        singular_value_floor (float): clip singular values of ``K_LL`` below
            this to this value before inverting (paper/original repo use
            ``1e-7``).
        batch_size (int): if set, ``kernel_fn`` is called on chunks of ``X``
            of this size instead of all at once (memory management for large
            data sets / expensive kernels).

    The output feature dimension is always exactly ``len(landmarks)`` --
    singular values are floored, never dropped, so ``q`` is preserved.
    """

    def __init__(self, landmarks, kernel_fn, singular_value_floor=1e-7, batch_size=None):
        self.landmarks = landmarks
        self.kernel_fn = kernel_fn
        self.batch_size = batch_size
        K_LL = np.asarray(kernel_fn(landmarks, landmarks))
        U, S, V = np.linalg.svd(K_LL)
        S = np.maximum(S, singular_value_floor)
        self._normalization = (U / np.sqrt(S)) @ V  # ~= K_LL^{-1/2} (K_LL is symmetric)

    @property
    def dim(self):
        return len(self.landmarks)

    def __call__(self, X):
        """``phi(X)``: an ``(n, q)`` array of explicit features for data ``X``."""
        if self.batch_size is None:
            K_XL = np.asarray(self.kernel_fn(X, self.landmarks))
            return (K_XL @ self._normalization.T).astype(np.float32)
        chunks = []
        for start in range(0, len(X), self.batch_size):
            K_XL = np.asarray(self.kernel_fn(X[start:start + self.batch_size], self.landmarks))
            chunks.append(K_XL @ self._normalization.T)
        return np.concatenate(chunks, axis=0).astype(np.float32)


def sample_landmarks(X, q, seed=0):
    """Uniformly sample ``q`` landmark rows from ``X`` for the Nystrom map.

    Returns ``(landmarks, indices)``.
    """
    rs = np.random.RandomState(seed)
    inds = rs.choice(len(X), size=min(q, len(X)), replace=False)
    return X[inds], inds
