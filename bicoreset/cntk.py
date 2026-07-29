"""6-layer Convolutional Neural Tangent Kernel (CNTK) with global average
pooling for CIFAR-10, Sec. 5.1.

The paper's Sec. 5.1 setup: "Our target model is multiclass logistic
regression, where the feature space is the q = 2048-dimensional Nystrom
feature space of the Convolutional Neural Tangent Kernel (CNTK) proposed by
Arora et al. (2019) with six layers and global average pooling on CIFAR-10."

The paper text does not specify the exact channel widths, strides or padding
of this CNTK beyond "six layers and global average pooling" and the citation
to Arora et al. (2019), "On Exact Computation with an Infinitely Wide Neural
Net" (NeurIPS 2019). This module reconstructs a standard CNTK-6 in that spirit
using ``neural_tangents.stax``: six 3x3 SAME-padded conv+ReLU blocks (spatial
size preserved throughout, no pooling between blocks -- consistent with the
"six layers" count referring to conv layers only, not pooling layers) followed
by global average pooling and a linear readout. If you have access to the
original repository's exact architecture, pass different ``channels``/
``depth``/``w_std``/``b_std`` to override.

Requires ``jax`` + ``neural-tangents`` (imported lazily -- this module can be
imported without them; only calling :func:`build_cntk6_gap_kernel_fn` needs
them). Also needs ``jax_patch`` imported first to hot-patch jax for the
``neural_tangents`` version pinned in this repo -- see ``cl.py``'s
``get_kernel_fn`` for the same convention.

Computing an NTK Gram matrix for CIFAR-10-sized (32x32x3) images with a 6-layer
conv kernel is expensive: unlike a plain forward pass, kernel evaluation scales
with the *depth* of the network for every pair of images, and is not
accelerated by the infinite-width limit the way training is -- there is no
shortcut around evaluating six conv layers' worth of kernel propagation for
every (landmark, image) pair. For the Nystrom feature map (Sec. 3.5.2) this
means O(q * n) kernel evaluations rather than O(n^2), which is what makes
Sec. 5.1 tractable at all, but with q=2048 landmarks and n up to 50000 CIFAR-10
images this can still take hours on a single GPU. Reduce ``q`` and/or the
number of images for a smoke test before committing to a full run.

**``diagonal_spatial`` is NOT passed at call time.** The architecture ends
with ``GlobalAvgPool``, which imposes ``Diagonal(input=NO, output=YES)`` --
full spatial covariance on input (needed to compute the average correctly),
diagonal on output. Passing ``diagonal_spatial=True`` (diagonal on *both*)
conflicts with this requirement and raises ``ValueError``. Omitting the kwarg
lets ``neural_tangents`` use the architecture's own setting automatically.
This means memory scales as O(n1 * n2 * H^2 * W^2) per chunk -- controlled
by ``batch_size`` (default 4, yielding ~614 MB peak for 28x28 images or
~1 GB for 32x32, feasible on T4 16GB). This is the *exact* CNTK kernel
(no diagonal approximation), matching the Arora et al. (2019) construction.
"""

import numpy as np


def build_cntk6_gap_kernel_fn(channels=64, depth=6, w_std=1.6, b_std=0.05,
                              batch_size=4, n_classes=10):
    """Build a batched, jit-compiled CNTK-6 + global-average-pooling kernel fn.

    Note on ``diagonal_spatial``: this function does NOT pass
    ``diagonal_spatial`` to ``kernel_fn`` -- the architecture ends with
    ``GlobalAvgPool``, which imposes ``Diagonal(input=NO, output=YES)``
    (full spatial covariance on input, diagonal on output). Passing
    ``diagonal_spatial=True`` conflicts with this and raises ``ValueError``.
    Omitting it lets ``neural_tangents`` use the architecture's own
    requirement. OOM is controlled by ``batch_size`` instead.

    Args:
        channels (int): number of channels per conv layer (does not change the
            NTK itself in the infinite-width limit, but is required to define
            the ``stax`` layers).
        depth (int): number of conv+ReLU blocks (paper: 6).
        w_std, b_std (float): weight/bias std of every layer -- these *do*
            affect the resulting kernel values (paper does not report the
            exact values it used; ``w_std=1.6`` roughly compensates for the
            variance-halving effect of ReLU as depth grows, ``b_std=0.05``
            matches the convention already used elsewhere in this repo's
            ``cl_streaming/ntk_generator.py``).
        batch_size (int): both ``X`` and ``Y`` are chunked to at most this many
            examples per single ``kernel_fn`` (jit) call -- memory/latency
            tradeoff, and critical for avoiding OOM. Without diagonal_spatial
            approximation (not available with GlobalAvgPool), full spatial
            covariance uses O(batch^2 * H^2 * W^2) per layer with large
            intermediate buffers across 6 conv layers. batch_size=4 keeps
            peak GPU memory under ~4 GB for 28x28 images on T4 16GB.
        n_classes (int): output dimension of the (unused) readout layer --
            only the kernel function is used, not the readout, but ``stax``
            needs a full network definition.

    Returns:
        callable: ``kernel_fn_np(X, Y) -> np.ndarray`` of shape
        ``(len(X), len(Y))``, X/Y being ``(n, 32, 32, 3)`` float arrays (NHWC,
        as CIFAR-10 is natively stored).
    """
    import jax_patch  # noqa: F401  (hot-patches jax; must be imported before neural_tangents)
    from jax import jit
    from neural_tangents import stax

    layers = []
    for _ in range(depth):
        layers += [stax.Conv(channels, (3, 3), (1, 1), padding='SAME', W_std=w_std, b_std=b_std)]
        layers += [stax.Relu()]
    layers += [stax.GlobalAvgPool()]
    layers += [stax.Dense(n_classes, w_std, b_std)]
    _, _, kernel_fn = stax.serial(*layers)
    # static_argnums=(2,) marks the positional 'get' arg (='ntk') static.
    # We do NOT pass diagonal_spatial to kernel_fn at call time: the architecture
    # ends with GlobalAvgPool, which imposes its own spatial-covariance requirement
    # (Diagonal(input=NO, output=YES)) -- passing diagonal_spatial=True conflicts
    # with that and raises ValueError. Omitting it lets neural_tangents use the
    # architecture's own requirement automatically.
    # diagonal_batch is still passed explicitly (=False) so we declare it static
    # to avoid TracerBoolConversionError (neural_tangents calls bool() on it
    # during jit tracing).
    kernel_fn = jit(kernel_fn, static_argnums=(2,),
                    static_argnames=('diagonal_batch',))

    def kernel_fn_np(X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        n, m = len(X), len(Y)
        K = np.zeros((n, m), dtype=np.float32)
        # Chunk *both* X and Y: a single kernel_fn(x1, x2) call materializes an
        # intermediate tensor scaling with len(x1) * len(x2) * H^2 * W^2
        # (full spatial covariance, required by GlobalAvgPool) -- leaving
        # either side unchunked still OOMs once a side exceeds a few hundred.
        # With batch_size=16 on 28x28 images: ~614 MB peak per chunk, feasible
        # on T4/P100. For CIFAR-10 32x32: ~1 GB, still OK at batch_size=16.
        for xi in range(0, n, batch_size):
            xj = min(xi + batch_size, n)
            for yi in range(0, m, batch_size):
                yj = min(yi + batch_size, m)
                K[xi:xj, yi:yj] = np.array(kernel_fn(
                    X[xi:xj], Y[yi:yj], 'ntk',
                    diagonal_batch=False))
        return K

    return kernel_fn_np
