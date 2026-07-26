"""Practical variants of Algorithm 1, Sec. 5.1, on real FashionMNIST (ConvNet).

This is the "real experiment" counterpart of ``demos/demo_algorithm1_variants.py``:
the same six methods (Uniform, BiCo Fwd, BiCo Fwd b, BiCo Elim, BiCo Exch,
BiCo Reg), but built directly on the ``models.ConvNet`` target model over the
actual FashionMNIST image data set already used elsewhere in this repo
(``cl_streaming/cl.py``, ``demos/demo_fashion_direct.py``), instead of the
``sklearn.datasets.load_digits`` substitute used for a fast CPU demo.

Unlike the demo script, this follows the runner/results/plotter pattern
already used by ``cl_streaming/cl.py`` + ``experiments/baseline_comparison_plot.py``:
run this once per ``(method, size, seed)``, it saves a JSON result file to
``--output-dir``; then run ``algorithm1_variants_fashion_plot.py`` to read
every saved file and produce the Figure-3-style plot.

Note on hyperparameters: the paper's Sec. 5.1 numbers (``b=1`` forward
selection, elimination in batches of 200, 200 exchange steps, 100 CG steps)
apply to a *linear* proxy -- logistic regression on precomputed 2048-d
Nystrom-CNTK features (that literal setup is what
``demos/demo_algorithm1_variants.py`` reproduces). Here the inner problem is
a full ConvNet retrained from scratch at every selection step
(``retrain_from_scratch=True``), orders of magnitude more expensive per
step; the defaults below (bounded candidate pool, fewer exchange steps,
Neumann-series IHVP instead of CG) follow the same practical choices already
validated in ``demos/demo_fashion_direct.py`` for GPU tractability -- they
are a structural necessity of using a real CNN as the target model, not an
arbitrary speed hack. Every one of them is a CLI flag: raise them on a
Kaggle GPU if you want a tighter approximation.

Usage -- run once per method/size/seed (loop over these from a shell script
or notebook cell; sizes are in absolute number of images):

    python experiments/run_algorithm1_variants_fashion.py --method full --seed 0
    python experiments/run_algorithm1_variants_fashion.py --method uniform    --size 200 --seed 0
    python experiments/run_algorithm1_variants_fashion.py --method bico_fwd   --size 200 --seed 0
    python experiments/run_algorithm1_variants_fashion.py --method bico_fwd_b --size 200 --seed 0
    python experiments/run_algorithm1_variants_fashion.py --method bico_elim  --size 200 --seed 0
    python experiments/run_algorithm1_variants_fashion.py --method bico_exch  --size 200 --seed 0
    python experiments/run_algorithm1_variants_fashion.py --method bico_reg   --size 200 --seed 0

Then:

    python experiments/algorithm1_variants_fashion_plot.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models
from bicoreset import losses
from bicoreset.direct import BilevelCoreset
from bicoreset.regularized import RegularizedBilevelCoreset

METHODS = ['uniform', 'bico_fwd', 'bico_fwd_b', 'bico_elim', 'bico_exch', 'bico_reg', 'full']


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_fashion_mnist(data_root='data'):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # FashionMNIST mean/std
    ])
    train = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
    return train, test


def train_convnet(dataset, nr_classes=10, nr_epochs=1000, batch_size=128, device='cpu', lr=5e-4):
    """Same training routine as demo_fashion_direct.py / demo_fashion.ipynb."""
    model = models.ConvNet(output_dim=nr_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    model.train()
    for _ in range(nr_epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
    return model


def evaluate(model, dataset, device='cpu', batch_size=512):
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            correct += (model(data).argmax(dim=1) == target).sum().item()
    return correct / len(dataset)


def fwd_builder(args, device):
    return BilevelCoreset(
        model_fn=lambda: models.ConvNet(output_dim=10),
        loss_fn=losses.cross_entropy,
        inner_reg=1e-4,
        ihvp='neumann', ihvp_kwargs={'num_terms': 50, 'alpha': 0.01, 'damping': 1e-3},
        max_inner_it=args.max_inner_it, inner_lr=5e-4,
        max_outer_it=0,  # binary weights, Sec. 3.5.1
        candidate_pool_size=args.candidate_per_step,
        outer_batch_size=256,
        hessian_batch_size=64,  # stochastic Hessian, Sec. 5.2.3
        retrain_from_scratch=True,
        device=device, verbose=args.verbose, logging_period=5)


def reg_builder(args, device):
    # Every call to build() starts from w = uniform 1/n over the *whole* candidate
    # pool (regardless of target size), so shrinking to a small target needs many
    # more beta-doublings -- and hence more outer iterations -- than shrinking to
    # a large one. With a small, fixed --reg-outer-it budget the small-size runs
    # can hit max_outer_it before beta/the support have actually converged and
    # fall back to "keep the heaviest points" truncation (regularized.py L271-275),
    # which produces a poorly-optimized (noisy) coreset -- the likely cause of the
    # BiCo Reg curve dipping non-monotonically. Raise --reg-outer-it and
    # --reg-warm-inner-it first if you see that; run more --seeds to see whether
    # the dip is a real trend or just single-run ConvNet training noise.
    return RegularizedBilevelCoreset(
        model_fn=lambda: models.ConvNet(output_dim=10),
        loss_fn=losses.cross_entropy,
        inner_reg=1e-4,
        beta=args.reg_beta, adaptive_beta=True, patience=args.reg_patience,
        max_outer_it=args.reg_outer_it,
        outer_lr=0.05,
        max_inner_it=args.max_inner_it, warm_inner_it=args.reg_warm_inner_it,
        ihvp='neumann', ihvp_kwargs={'num_terms': 50, 'alpha': 0.01, 'damping': 1e-3},
        retrain_from_scratch=args.reg_retrain_from_scratch,
        device=device, logging_period=10_000, verbose=args.verbose)


def build_indices(method, X_pool, y_pool, size, seed, args, device):
    """Returns (indices into the pool, weights or None, build_time)."""
    rs = np.random.RandomState(seed)
    n_pool = X_pool.shape[0]
    start = min(10, size)

    if method == 'uniform':
        t0 = time.time()
        inds = rs.choice(n_pool, size, replace=False)
        return inds, None, time.time() - t0

    if method == 'bico_fwd':
        t0 = time.time()
        builder = fwd_builder(args, device)
        inds, w = builder.build(X_pool, y_pool, size, strategy='forward',
                                selection_batch_size=1, start_size=start)
        return inds, None, time.time() - t0

    if method == 'bico_fwd_b':
        t0 = time.time()
        builder = fwd_builder(args, device)
        inds, w = builder.build(X_pool, y_pool, size, strategy='forward',
                                selection_batch_size=args.fwd_b_batch, start_size=start)
        return inds, None, time.time() - t0

    if method == 'bico_elim':
        t0 = time.time()
        builder = fwd_builder(args, device)
        inds, w = builder.build(X_pool, y_pool, size, strategy='elimination',
                                selection_batch_size=args.elim_batch)
        return inds, None, time.time() - t0

    if method == 'bico_exch':
        t0 = time.time()
        builder = fwd_builder(args, device)
        exch_batch = max(1, int(round(0.01 * size)))  # paper: 1% of the selected points per step
        inds, w = builder.build(X_pool, y_pool, size, strategy='exchange',
                                selection_batch_size=exch_batch, n_exchange_steps=args.exch_steps)
        return inds, None, time.time() - t0

    if method == 'bico_reg':
        t0 = time.time()
        builder = reg_builder(args, device)
        inds, w = builder.build(X_pool, y_pool, m=size)
        return inds, w, time.time() - t0

    raise ValueError('unknown method "{}"'.format(method))


def run_full(train_dataset, test_dataset, seed, args, device, output_dir):
    """Train on the *entire* training set -- the "Full Dataset" reference line."""
    set_seed(seed)
    t0 = time.time()
    net = train_convnet(train_dataset, nr_epochs=args.full_epochs,
                        batch_size=args.batch_size, device=device)
    train_time = time.time() - t0
    acc = evaluate(net, test_dataset, device=device)
    result = {'method': 'full', 'size': len(train_dataset), 'seed': seed,
             'test_acc': 100.0 * acc, 'build_time': 0.0, 'train_time': train_time}
    _save(output_dir, 'full_{}.txt'.format(seed), result)
    print('full dataset (n={}): {:.2f}%  [train {:.1f}s]'.format(len(train_dataset), 100 * acc, train_time))


def _save(output_dir, filename, result):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(result, f)


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    set_seed(args.seed)

    train_dataset, test_dataset = load_fashion_mnist(args.data_root)
    print('training set: {} images, test set: {} images'.format(len(train_dataset), len(test_dataset)))

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'algo1_fashion_results')

    if args.method == 'full':
        run_full(train_dataset, test_dataset, args.seed, args, device, output_dir)
        return

    if args.size is None:
        raise ValueError('--size is required for method "{}"'.format(args.method))

    # Restrict the candidate pool the selection algorithms get to see (same
    # tractability trick as demos/demo_fashion_direct.py's `limit`), sampled
    # once per seed so every method/size sees an identical pool.
    set_seed(args.seed)
    pool_perm = np.random.RandomState(args.seed).choice(
        len(train_dataset), min(args.candidate_pool, len(train_dataset)), replace=False)
    loader = DataLoader(Subset(train_dataset, pool_perm), batch_size=len(pool_perm), shuffle=False)
    X_pool, y_pool = next(iter(loader))

    set_seed(args.seed)
    inds_in_pool, weights, build_time = build_indices(
        args.method, X_pool, y_pool, args.size, args.seed, args, device)
    dataset_inds = pool_perm[inds_in_pool]

    t0 = time.time()
    net = train_convnet(Subset(train_dataset, dataset_inds), nr_epochs=args.train_epochs,
                        batch_size=args.batch_size, device=device)
    train_time = time.time() - t0
    acc = evaluate(net, test_dataset, device=device)

    result = {'method': args.method, 'size': args.size, 'seed': args.seed,
             'test_acc': 100.0 * acc, 'build_time': build_time, 'train_time': train_time,
             'candidate_pool': len(pool_perm)}
    _save(output_dir, '{}_{}_{}.txt'.format(args.method, args.size, args.seed), result)
    print('{} (m={}, seed={}): {:.2f}%  [build {:.1f}s, train {:.1f}s]'.format(
        args.method, args.size, args.seed, 100 * acc, build_time, train_time))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--size', type=int, default=None, help='coreset size (required unless --method full)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--candidate-pool', type=int, default=2000,
                        help='number of training images the selection algorithms get to see')
    parser.add_argument('--candidate-per-step', type=int, default=300,
                        help='candidates scored per forward-selection step (bicoreset.direct candidate_pool_size)')
    parser.add_argument('--fwd-b-batch', type=int, default=10, help='batch size for "bico_fwd_b"')
    parser.add_argument('--elim-batch', type=int, default=100, help='elimination batch size for "bico_elim"')
    parser.add_argument('--exch-steps', type=int, default=20,
                        help='exchange rounds for "bico_exch" (paper: 200; see module docstring)')
    parser.add_argument('--reg-outer-it', type=int, default=150,
                        help='outer iterations for "bico_reg" (Algorithm 2); raise this first if the '
                             'BiCo Reg curve looks noisy/non-monotonic -- small target sizes need many '
                             'more beta-doublings to converge within the budget than large ones (paper: 200)')
    parser.add_argument('--reg-warm-inner-it', type=int, default=30,
                        help='inner GD steps to re-fit the ConvNet after each weight update in "bico_reg" '
                             '(raise this together with --reg-outer-it if the curve is noisy)')
    parser.add_argument('--reg-patience', type=int, default=3,
                        help='plateau length (in outer iterations) before beta is doubled in "bico_reg"')
    parser.add_argument('--reg-beta', type=float, default=1e-6,
                        help='initial sparsity penalty for "bico_reg" (paper starts at 1e-7)')
    parser.add_argument('--reg-retrain-from-scratch', action='store_true',
                        help='fully retrain the ConvNet from scratch after every "bico_reg" weight update '
                             'instead of warm-starting (much slower, but removes warm-start staleness -- '
                             'try this if raising --reg-outer-it/--reg-warm-inner-it is not enough)')
    parser.add_argument('--max-inner-it', type=int, default=300,
                        help='inner GD steps to fit the ConvNet on the current support')
    parser.add_argument('--train-epochs', type=int, default=1000,
                        help='epochs used to train the final ConvNet on the selected coreset')
    parser.add_argument('--full-epochs', type=int, default=30,
                        help='epochs used to train the "full dataset" reference ConvNet')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default=None, choices=[None, 'cpu', 'cuda'])
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
