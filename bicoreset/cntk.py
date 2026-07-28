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
"""

import numpy as np


def build_cntk6_gap_kernel_fn(channels=64, depth=6, w_std=1.6, b_std=0.05,
                              batch_size=64, n_classes=10):
    """Build a batched, jit-compiled CNTK-6 + global-average-pooling kernel fn.

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
        batch_size (int): number of landmark columns scored per jit call
            (memory/latency tradeoff, mirrors ``generate_cnn_ntk``).
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
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    def kernel_fn_np(X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        n, m = len(X), len(Y)
        K = np.zeros((n, m), dtype=np.float32)
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            K[:, start:end] = np.array(kernel_fn(X, Y[start:end], 'ntk'))
        return K

    return kernel_fn_np
