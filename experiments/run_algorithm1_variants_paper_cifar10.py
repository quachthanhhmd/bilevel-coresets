"""Literal reproduction of Sec. 5.1 / Figure 3 on real CIFAR-10, plus the same
CNTK-Nystrom-proxy + logistic-regression *methodology* applied to FashionMNIST
via ``--dataset fashionmnist``.

Unlike ``run_algorithm1_variants_fashion.py`` (a ConvNet trained directly on
FashionMNIST -- a *different*, harder setting the paper does not use for
Sec. 5.1), this script uses the paper's actual Sec. 5.1 setup as closely as
the paper text specifies:

With ``--dataset cifar10`` (default) this is a literal reproduction of Fig. 3
-- every hyperparameter below is the paper's own value. With ``--dataset
fashionmnist`` the *same* CNTK-Nystrom-proxy + logistic-regression pipeline
(same hyperparameters, same code path) is applied to FashionMNIST instead --
this is **not** a paper number, it is "does the paper's Sec. 5.1 methodology,
unmodified, transfer to a different dataset", complementary to (not a
replacement for) the CIFAR-10 run. Report the two separately; don't present
FashionMNIST numbers from this script as reproducing Fig. 3.

* Target model: multiclass logistic regression on the q=2048-dimensional
  Nystrom feature space of a 6-layer CNTK with global average pooling
  (:mod:`bicoreset.cntk`, :mod:`bicoreset.nystrom`) -- *not* a CNN trained
  directly on pixels.
* Data: real CIFAR-10, split into a 90% train / 10% validation partition of
  the original 50000-image training set (paper, Sec. 5.1); the outer
  objective sums train and validation losses, the coreset is selected from
  the train partition only; test accuracy is measured on the real CIFAR-10
  test set (10000 images).
* Hyperparameters straight from Appendix C ("Variants" + "Neural Networks" is
  *not* used here -- Sec. 5.1 is not a deep net): inner_reg (lambda) = 1e-7,
  Adam inner optimizer with step size 0.01, warm-start with 5e4 initial +
  1e4 GD iterations per subsequent selection step (``--first-inner-it``/
  ``--max-inner-it``), implicit gradients via 100 conjugate-gradient steps.
* Variants (Sec. 5.1 main text): one-by-one forward selection (b=1),
  forward selection in batches of 25 ("BiCo Fwd 25"), elimination in batches
  of 200, exchange with 200 steps (1% of the selected points per step), and
  the regularized/weighted variant of Algorithm 2 ("BiCo Reg").

Coreset sizes are given as a *percentage* of the train partition (45000
images for CIFAR-10, 54000 for FashionMNIST's 90/10 split of 60000), matching
Figure 3's x-axis (paper ticks: 0.5%, 2%, 8%, 32%, 100%).

IMPORTANT -- compute cost: computing the CNTK-Nystrom feature map for the
full train partition (q=2048 landmarks, 6-layer conv kernel) is very
expensive -- kernel evaluation cost scales with network depth for *every*
(landmark, image) pair, with no infinite-width shortcut the way training has.
Expect this to take a long time even on a Kaggle GPU (FashionMNIST's smaller
28x28x1 images are somewhat cheaper per pair than CIFAR-10's 32x32x3, but the
scaling is otherwise the same). Features are cached to ``--features-cache-dir``
after the first computation (keyed by dataset/seed/q/kernel hyperparameters)
and reused by every subsequent method/size/seed run, which is the only way
this is tractable to run more than once.

IMPORTANT -- memory: by default the CNTK kernel uses ``diagonal_spatial=True``
(see :mod:`bicoreset.cntk`) and chunks both sides of every kernel_fn call to
``--kernel-batch-size`` (default 16). Without this, a real run OOM'd on a
Kaggle GPU trying to allocate >30GB for a single landmark batch -- the exact
(non-diagonal-spatial) CNTK is essentially infeasible for CIFAR-10-sized
images on typical single-GPU hardware. Don't pass ``--cntk-exact-spatial``
unless you know your GPU has enough memory.

Before committing to a full run, do a smoke test with small overrides, e.g.:

    python experiments/run_algorithm1_variants_paper_cifar10.py --method full \\
        --dataset fashionmnist --seed 0 --nystrom-dim 128 --train-pool-size 2000 \\
        --val-size 200 --first-inner-it 200 --max-inner-it 50

These overrides are *not* defaults -- the defaults below are the paper's
literal Sec. 5.1 values regardless of ``--dataset`` (per the instruction to
keep this reproduction faithful and let heavy runs happen on Kaggle, not to
silently scale things down).

Requires jax + neural-tangents (see :mod:`bicoreset.cntk`) and internet
access to download the dataset on first run.

Usage (run once per method/size/seed, same pattern as
``run_algorithm1_variants_fashion.py``; add ``--dataset fashionmnist`` to any
of these to switch data sets, default is ``cifar10``):

    python experiments/run_algorithm1_variants_paper_cifar10.py --method full --seed 0
    python experiments/run_algorithm1_variants_paper_cifar10.py --method uniform    --size-pct 0.5 --seed 0
    python experiments/run_algorithm1_variants_paper_cifar10.py --method bico_fwd   --size-pct 0.5 --seed 0
    python experiments/run_algorithm1_variants_paper_cifar10.py --method bico_fwd25 --size-pct 0.5 --seed 0
    python experiments/run_algorithm1_variants_paper_cifar10.py --method bico_elim  --size-pct 0.5 --seed 0
    python experiments/run_algorithm1_variants_paper_cifar10.py --method bico_exch  --size-pct 0.5 --seed 0
    python experiments/run_algorithm1_variants_paper_cifar10.py --method bico_reg   --size-pct 0.5 --seed 0

Then:

    python experiments/algorithm1_variants_paper_cifar10_plot.py --dataset fashionmnist
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bicoreset import losses
from bicoreset.direct import BilevelCoreset
from bicoreset.regularized import RegularizedBilevelCoreset
from bicoreset.nystrom import NystromFeatureMap, sample_landmarks

METHODS = ['uniform', 'bico_fwd', 'bico_fwd25', 'bico_elim', 'bico_exch', 'bico_reg', 'full']


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_cifar10(data_root='data'):
    """Returns ``(train_images, train_labels, test_images, test_labels)`` as
    ``float32`` NHWC arrays in ``[0, 1]`` / ``int64`` label arrays."""
    import torchvision.datasets as datasets

    train = datasets.CIFAR10(root=data_root, train=True, download=True)
    test = datasets.CIFAR10(root=data_root, train=False, download=True)
    X_train = (train.data.astype(np.float32) / 255.0)  # (50000, 32, 32, 3), already NHWC
    y_train = np.array(train.targets, dtype=np.int64)
    X_test = (test.data.astype(np.float32) / 255.0)
    y_test = np.array(test.targets, dtype=np.int64)
    return X_train, y_train, X_test, y_test


def load_fashion_mnist_raw(data_root='data'):
    """Returns ``(train_images, train_labels, test_images, test_labels)`` as
    ``float32`` NHWC (n, 28, 28, 1) arrays in ``[0, 1]`` / ``int64`` labels --
    same format as :func:`load_cifar10` so the rest of the pipeline (CNTK
    kernel, Nystrom map, logistic regression) is unchanged. Raw pixels, no
    Normalize (the CNTK/Nystrom construction has no notion of the
    ``(0.2860, 0.3530)`` normalization used elsewhere in this repo for
    ConvNet training; kernel methods are typically applied to raw or
    per-image-standardized inputs instead)."""
    import torchvision.datasets as datasets

    train = datasets.FashionMNIST(root=data_root, train=True, download=True)
    test = datasets.FashionMNIST(root=data_root, train=False, download=True)
    X_train = (train.data.numpy().astype(np.float32) / 255.0)[..., None]  # (60000, 28, 28, 1)
    y_train = train.targets.numpy().astype(np.int64)
    X_test = (test.data.numpy().astype(np.float32) / 255.0)[..., None]
    y_test = test.targets.numpy().astype(np.int64)
    return X_train, y_train, X_test, y_test


def load_dataset(name, data_root='data'):
    if name == 'cifar10':
        return load_cifar10(data_root)
    if name == 'fashionmnist':
        return load_fashion_mnist_raw(data_root)
    raise ValueError('unknown dataset "{}"'.format(name))


def split_train_val(X, y, val_frac=0.1, seed=0):
    """90/10 train/val split of the original training set (Sec. 5.1's protocol,
    applied here to whichever dataset was loaded)."""
    rs = np.random.RandomState(seed)
    n = len(X)
    perm = rs.permutation(n)
    n_val = int(round(val_frac * n))
    val_inds, train_inds = perm[:n_val], perm[n_val:]
    return X[train_inds], y[train_inds], X[val_inds], y[val_inds]


# ----------------------------------------------------------------------
# CNTK-Nystrom features (cached -- this is the expensive part)
# ----------------------------------------------------------------------
def _cache_key(args):
    payload = 'ds={}_q={}_ch={}_depth={}_wstd={}_bstd={}_exact={}_seed={}_pool={}_val={}'.format(
        args.dataset, args.nystrom_dim, args.cntk_channels, args.cntk_depth, args.cntk_w_std,
        args.cntk_b_std, args.cntk_exact_spatial, args.seed, args.train_pool_size, args.val_size)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def compute_or_load_features(args):
    """Returns ``(Phi_train, y_train, Phi_val, y_val, Phi_test, y_test)``.

    ``Phi_*`` are ``(n, q)`` Nystrom-CNTK features. Cached to
    ``--features-cache-dir`` keyed by the kernel/seed hyperparameters so every
    method/size/seed run after the first reuses them instead of recomputing
    the (very expensive) kernel.
    """
    cache_dir = args.features_cache_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '{}_cntk_features'.format(args.dataset))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'features_{}.npz'.format(_cache_key(args)))

    if os.path.exists(cache_path):
        print('Loading cached CNTK-Nystrom features from {}'.format(cache_path))
        data = np.load(cache_path)
        return (data['Phi_train'], data['y_train'], data['Phi_val'], data['y_val'],
                data['Phi_test'], data['y_test'])

    print('No cached features at {} -- computing (this is the expensive step)'.format(cache_path))
    from bicoreset.cntk import build_cntk6_gap_kernel_fn

    X_train_full, y_train_full, X_test, y_test = load_dataset(args.dataset, args.data_root)
    if args.val_size is None:
        X_train, y_train, X_val, y_val = split_train_val(X_train_full, y_train_full, seed=args.seed)
    else:
        # smoke-test override: fixed-size train/val subsample instead of a 90/10 split
        rs = np.random.RandomState(args.seed)
        perm = rs.permutation(len(X_train_full))
        train_inds = perm[:args.train_pool_size or (len(X_train_full) - args.val_size)]
        val_inds = perm[-args.val_size:]
        X_train, y_train = X_train_full[train_inds], y_train_full[train_inds]
        X_val, y_val = X_train_full[val_inds], y_train_full[val_inds]
    if args.train_pool_size is not None and args.val_size is None:
        X_train, y_train = X_train[:args.train_pool_size], y_train[:args.train_pool_size]

    print('train pool: {}, val: {}, test: {}'.format(len(X_train), len(X_val), len(X_test)))

    kernel_fn = build_cntk6_gap_kernel_fn(
        channels=args.cntk_channels, depth=args.cntk_depth,
        w_std=args.cntk_w_std, b_std=args.cntk_b_std,
        batch_size=args.kernel_batch_size,
        diagonal_spatial=not args.cntk_exact_spatial)

    landmarks, _ = sample_landmarks(X_train, args.nystrom_dim, seed=args.seed)
    print('Fitting Nystrom map with {} landmarks...'.format(len(landmarks)))
    t0 = time.time()
    phi = NystromFeatureMap(landmarks, kernel_fn, batch_size=args.kernel_batch_size)
    print('Landmark Gram matrix done in {:.1f}s'.format(time.time() - t0))

    def featurize(X, name):
        t0 = time.time()
        F = phi(X)
        print('  featurized {} ({} points) in {:.1f}s'.format(name, len(X), time.time() - t0))
        return F

    Phi_train = featurize(X_train, 'train')
    Phi_val = featurize(X_val, 'val')
    Phi_test = featurize(X_test, 'test')

    np.savez(cache_path, Phi_train=Phi_train, y_train=y_train, Phi_val=Phi_val, y_val=y_val,
             Phi_test=Phi_test, y_test=y_test)
    print('Cached features to {}'.format(cache_path))
    return Phi_train, y_train, Phi_val, y_val, Phi_test, y_test


# ----------------------------------------------------------------------
# Target model + evaluation
# ----------------------------------------------------------------------
def make_logreg(q, n_classes=10):
    return nn.Linear(q, n_classes)


def train_logreg(Phi, y, q, n_classes=10, n_steps=50_000, lr=0.01, device='cpu'):
    model = make_logreg(q, n_classes).to(device)
    X = torch.from_numpy(Phi).float().to(device)
    Y = torch.from_numpy(y).long().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X), Y)
        loss.backward()
        optimizer.step()
    return model


def evaluate(model, Phi, y, device='cpu'):
    model = model.to(device).eval()
    X = torch.from_numpy(Phi).float().to(device)
    Y = torch.from_numpy(y).long().to(device)
    with torch.no_grad():
        acc = (model(X).argmax(dim=1) == Y).float().mean().item()
    return acc


# ----------------------------------------------------------------------
# Coreset construction (Sec. 5.1 variants)
# ----------------------------------------------------------------------
def fwd_builder(args, q, device):
    return BilevelCoreset(
        model_fn=lambda: make_logreg(q),
        loss_fn=losses.cross_entropy,
        inner_reg=args.inner_reg,
        ihvp='cg', ihvp_kwargs={'max_iter': args.cg_iters},
        inner_lr=args.inner_lr,
        first_inner_it=args.first_inner_it, max_inner_it=args.max_inner_it,
        max_outer_it=0,  # binary weights, Sec. 3.5.1
        candidate_pool_size=args.candidate_per_step,  # None (paper default): score the entire remaining pool
        retrain_from_scratch=False,  # Appendix C "Variants": warm-start, not retrain from scratch
        device=device, verbose=args.verbose, logging_period=1)


def reg_builder(args, q, device):
    return RegularizedBilevelCoreset(
        model_fn=lambda: make_logreg(q),
        loss_fn=losses.cross_entropy,
        inner_reg=args.inner_reg,
        beta=args.reg_beta, adaptive_beta=True, patience=args.reg_patience,
        max_outer_it=args.reg_outer_it,
        outer_lr=0.01,
        max_inner_it=args.first_inner_it, warm_inner_it=args.max_inner_it,
        inner_lr=args.inner_lr,
        ihvp='cg', ihvp_kwargs={'max_iter': args.cg_iters},
        retrain_from_scratch=False,
        device=device, logging_period=5, verbose=args.verbose)


def build_indices(method, Phi_train, y_train, Phi_outer, y_outer, size, seed, args, device):
    rs = np.random.RandomState(seed)
    n_pool = Phi_train.shape[0]
    q = Phi_train.shape[1]

    if method == 'uniform':
        t0 = time.time()
        inds = rs.choice(n_pool, size, replace=False)
        return inds, None, time.time() - t0

    if method == 'bico_fwd':
        t0 = time.time()
        builder = fwd_builder(args, q, device)
        inds, w = builder.build(Phi_train, y_train, size, X_outer=Phi_outer, y_outer=y_outer,
                                strategy='forward', selection_batch_size=1, start_size=1)
        return inds, None, time.time() - t0

    if method == 'bico_fwd25':
        t0 = time.time()
        builder = fwd_builder(args, q, device)
        inds, w = builder.build(Phi_train, y_train, size, X_outer=Phi_outer, y_outer=y_outer,
                                strategy='forward', selection_batch_size=args.fwd25_batch, start_size=1)
        return inds, None, time.time() - t0

    if method == 'bico_elim':
        t0 = time.time()
        builder = fwd_builder(args, q, device)
        inds, w = builder.build(Phi_train, y_train, size, X_outer=Phi_outer, y_outer=y_outer,
                                strategy='elimination', selection_batch_size=args.elim_batch)
        return inds, None, time.time() - t0

    if method == 'bico_exch':
        t0 = time.time()
        builder = fwd_builder(args, q, device)
        exch_batch = max(1, int(round(0.01 * size)))  # paper: 1% of the selected points per step
        inds, w = builder.build(Phi_train, y_train, size, X_outer=Phi_outer, y_outer=y_outer,
                                strategy='exchange', selection_batch_size=exch_batch,
                                n_exchange_steps=args.exch_steps)
        return inds, None, time.time() - t0

    if method == 'bico_reg':
        t0 = time.time()
        builder = reg_builder(args, q, device)
        inds, w = builder.build(Phi_train, y_train, m=size, X_outer=Phi_outer, y_outer=y_outer)
        return inds, w, time.time() - t0

    raise ValueError('unknown method "{}"'.format(method))


# ----------------------------------------------------------------------
def _save(output_dir, filename, result):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(result, f)


def run_full(Phi_train, y_train, Phi_val, y_val, Phi_test, y_test, seed, args, device, output_dir):
    """"Full Dataset" reference: logistic regression on the entire train pool
    (the partition coresets are selected from), evaluated on the real test set
    of whichever dataset was loaded."""
    set_seed(seed)
    q = Phi_train.shape[1]
    t0 = time.time()
    model = train_logreg(Phi_train, y_train, q, n_steps=args.first_inner_it, lr=args.inner_lr, device=device)
    train_time = time.time() - t0
    acc = evaluate(model, Phi_test, y_test, device=device)
    result = {'method': 'full', 'dataset': args.dataset, 'size': len(Phi_train), 'seed': seed,
             'test_acc': 100.0 * acc, 'build_time': 0.0, 'train_time': train_time}
    _save(output_dir, 'full_{}.txt'.format(seed), result)
    print('full dataset (n={}): {:.2f}%  [train {:.1f}s]'.format(len(Phi_train), 100 * acc, train_time))


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    set_seed(args.seed)

    Phi_train, y_train, Phi_val, y_val, Phi_test, y_test = compute_or_load_features(args)
    q = Phi_train.shape[1]
    Phi_outer = np.concatenate([Phi_train, Phi_val], axis=0)  # outer loss = train + val losses (Sec. 5.1)
    y_outer = np.concatenate([y_train, y_val], axis=0)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'algo1_paper_{}_results'.format(args.dataset))

    if args.method == 'full':
        run_full(Phi_train, y_train, Phi_val, y_val, Phi_test, y_test, args.seed, args, device, output_dir)
        return

    if args.size_pct is None:
        raise ValueError('--size-pct is required for method "{}"'.format(args.method))
    size = max(1, int(round(args.size_pct / 100.0 * len(Phi_train))))

    set_seed(args.seed)
    inds, weights, build_time = build_indices(
        args.method, Phi_train, y_train, Phi_outer, y_outer, size, args.seed, args, device)

    t0 = time.time()
    model = train_logreg(Phi_train[inds], y_train[inds], q, n_steps=args.first_inner_it,
                         lr=args.inner_lr, device=device)
    train_time = time.time() - t0
    acc = evaluate(model, Phi_test, y_test, device=device)

    result = {'method': args.method, 'dataset': args.dataset, 'size': size, 'size_pct': args.size_pct,
             'seed': args.seed, 'test_acc': 100.0 * acc, 'build_time': build_time, 'train_time': train_time,
             'train_pool': len(Phi_train)}
    _save(output_dir, '{}_{}_{}.txt'.format(args.method, size, args.seed), result)
    print('{} (m={}, {}%, seed={}): {:.2f}%  [build {:.1f}s, train {:.1f}s]'.format(
        args.method, size, args.size_pct, args.seed, 100 * acc, build_time, train_time))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--dataset', choices=['cifar10', 'fashionmnist'], default='cifar10',
                        help='cifar10 (default): literal Sec. 5.1 reproduction, every default '
                             'below is a paper value. fashionmnist: same CNTK-Nystrom-proxy + '
                             'logistic-regression pipeline applied to FashionMNIST instead -- '
                             'not a paper number, report it separately from the cifar10 run')
    parser.add_argument('--size-pct', type=float, default=None,
                        help='coreset size as a %% of the train partition, e.g. 0.5, 2, 8, 32 '
                             '(paper Fig. 3 x-axis ticks). Required unless --method full')
    parser.add_argument('--seed', type=int, default=0)

    # CNTK-Nystrom features
    parser.add_argument('--nystrom-dim', type=int, default=2048, help='q, paper: 2048')
    parser.add_argument('--cntk-channels', type=int, default=64)
    parser.add_argument('--cntk-depth', type=int, default=6, help='paper: 6 ("six layers")')
    parser.add_argument('--cntk-w-std', type=float, default=1.6)
    parser.add_argument('--cntk-b-std', type=float, default=0.05)
    parser.add_argument('--cntk-exact-spatial', action='store_true',
                        help='use the exact (non-diagonal) spatial covariance for the CNTK conv '
                             'layers instead of the diagonal_spatial=True approximation. This is '
                             'closer to an exact CNTK but O(H^2*W^2) instead of O(H*W) in memory -- '
                             'a single 32x32 pair already needs >30GB and reliably OOMs on typical '
                             'single-GPU hardware (confirmed on a real Kaggle run). Leave off unless '
                             'you know your GPU has enough memory for this')
    parser.add_argument('--kernel-batch-size', type=int, default=16,
                        help='both X and Y are chunked to at most this many examples per kernel_fn '
                             'call (memory/latency tradeoff). Lowered from 64 to 16 by default after '
                             'a real OOM at 64/128 with the exact-spatial kernel; the default '
                             'diagonal_spatial=True kernel needs much less memory so you can likely '
                             'raise this back up once --cntk-exact-spatial is off (the default)')
    parser.add_argument('--train-pool-size', type=int, default=None,
                        help='override: subsample the train partition to this many images '
                             '(smoke-test only; paper uses the full ~45000-image partition)')
    parser.add_argument('--val-size', type=int, default=None,
                        help='override: use a fixed-size val subsample instead of the paper\'s '
                             '90/10 split (smoke-test only)')
    parser.add_argument('--features-cache-dir', default=None,
                        help='where to cache the (expensive) Nystrom-CNTK features; reused '
                             'across all method/size/seed runs with the same hyperparameters')

    # Inner problem (Appendix C "Variants")
    parser.add_argument('--inner-reg', type=float, default=1e-7, help='lambda, Eq. (8) (paper: 1e-7)')
    parser.add_argument('--inner-lr', type=float, default=0.01, help='Adam step size (paper: 0.01)')
    parser.add_argument('--first-inner-it', type=int, default=50_000,
                        help='GD steps on the initial point set (paper: 5e4)')
    parser.add_argument('--max-inner-it', type=int, default=10_000,
                        help='GD steps after each subsequent selection step (paper: 1e4)')
    parser.add_argument('--cg-iters', type=int, default=100, help='conjugate-gradient IHVP steps (paper: 100)')
    parser.add_argument('--candidate-per-step', type=int, default=None,
                        help='if set, only this many candidates are scored per selection step '
                             'instead of the entire remaining pool (paper does not subsample here; '
                             'set this for tractability if full-pool scoring is too slow)')

    # Variant-specific
    parser.add_argument('--fwd25-batch', type=int, default=25, help='paper: 25')
    parser.add_argument('--elim-batch', type=int, default=200, help='paper: 200')
    parser.add_argument('--exch-steps', type=int, default=200, help='paper: 200 (1%% of selected points/step)')
    parser.add_argument('--reg-beta', type=float, default=1e-7, help='initial sparsity penalty (paper: 1e-7)')
    parser.add_argument('--reg-outer-it', type=int, default=300,
                        help='outer iterations for "bico_reg" (Algorithm 2). The paper does not report '
                             'a fixed count for Sec. 5.1 -- see run_algorithm1_variants_fashion.py\'s '
                             '--reg-outer-it help for the same caveat; this is a practical starting point')
    parser.add_argument('--reg-patience', type=int, default=5, help='plateau length before beta doubles')

    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default=None, choices=[None, 'cpu', 'cuda'])
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
