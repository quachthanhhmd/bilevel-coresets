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
    # retrain_from_scratch is now a single shared flag (see --retrain-from-scratch)
    # instead of being hardcoded True here while reg_builder had its own separate
    # flag defaulting to False -- that split was inconsistent and didn't track any
    # real distinction the paper makes. Appendix C ("Variants") warm-starts *every*
    # Sec. 3.5 variant (5e4 initial Adam steps, then 1e4 more per selection step);
    # the paper only retrains from scratch for the Sec. 5.2.3 deep-net experiments
    # (WideResNet with a cosine-annealed LR schedule, Sec. 5.7). Our ConvNet with a
    # fixed Adam lr is arguably closer to the "Variants" setting, so we default to
    # warm-starting like the paper -- pass --retrain-from-scratch to switch to the
    # Sec. 5.2.3-style behavior if warm-starting looks unstable for a given method.
    return BilevelCoreset(
        model_fn=lambda: models.ConvNet(output_dim=10),
        loss_fn=losses.cross_entropy,
        inner_reg=args.inner_reg,
        ihvp='neumann', ihvp_kwargs={'num_terms': 50, 'alpha': 0.01, 'damping': 1e-3},
        max_inner_it=args.max_inner_it, inner_lr=5e-4,
        max_outer_it=0,  # binary weights, Sec. 3.5.1
        candidate_pool_size=args.candidate_per_step,
        outer_batch_size=256,
        hessian_batch_size=64,  # stochastic Hessian, Sec. 5.2.3
        retrain_from_scratch=args.retrain_from_scratch,
        device=device, verbose=args.verbose, logging_period=5)


def reg_builder(args, device):
    # Every call to build() starts from w = uniform 1/n over the *whole* candidate
    # pool (regardless of target size), so shrinking to a small target needs many
    # more beta-doublings -- and hence more outer iterations -- than shrinking to
    # a large one. With a small, fixed --reg-outer-it budget the small-size runs
    # can hit max_outer_it before beta/the support have actually converged and
    # fall back to "keep the heaviest points" truncation (regularized.py L271-275),
    # which produces a poorly-optimized (noisy) coreset.
    #
    # Note this is *not* purely an artifact of our ConvNet reproduction: the paper
    # itself reports non-monotonic BiCo Reg behavior in Fig. 3 ("the higher test
    # performance for the weighted coreset with size 20% compared to 90% is due to
    # the higher number of total outer gradient steps performed"), i.e. Sec. 5.1
    # also runs Algorithm 2 under a fixed *total* outer-step budget rather than a
    # fixed per-size --reg-outer-it. The paper does not report a specific outer
    # iteration count for BiCo Reg in Sec. 5.1 (the "150 outer iterations" figure
    # in Appendix C is for the *binary* logistic regression experiment of Sec.
    # 5.2.2, and the "200" figures in Sec. 5.1's text are BiCo Elim's batch size /
    # BiCo Exch's step count, not this). Our default below is just a starting
    # point for a real ConvNet -- raise --reg-outer-it / --reg-warm-inner-it if
    # the curve looks noisy, and run more --seeds to separate a real trend from
    # single-run ConvNet training noise.
    return RegularizedBilevelCoreset(
        model_fn=lambda: models.ConvNet(output_dim=10),
        loss_fn=losses.cross_entropy,
        inner_reg=args.inner_reg,
        beta=args.reg_beta, adaptive_beta=True, patience=args.reg_patience,
        max_outer_it=args.reg_outer_it,
        outer_lr=0.05,
        max_inner_it=args.max_inner_it, warm_inner_it=args.reg_warm_inner_it,
        ihvp='neumann', ihvp_kwargs={'num_terms': 50, 'alpha': 0.01, 'damping': 1e-3},
        retrain_from_scratch=args.retrain_from_scratch,
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
                        help='number of training images the selection algorithms get to see. Sec. 5.1 has '
                             'no analogous subsampling -- it scores implicit gradients over the *entire* '
                             'remaining CIFAR-10 training set (cheap: 2048-d Nystrom features + logistic '
                             'regression). The closest paper number is the pool of 2500 in Sec. 5.2.3\'s '
                             'WideResNet experiments, a different (real deep-net) setting; this default '
                             'is a practical choice for ConvNet+FashionMNIST tractability, not a Sec. 5.1 value')
    parser.add_argument('--candidate-per-step', type=int, default=300,
                        help='candidates scored per forward-selection step (bicoreset.direct candidate_pool_size)')
    parser.add_argument('--fwd-b-batch', type=int, default=10, help='batch size for "bico_fwd_b"')
    parser.add_argument('--elim-batch', type=int, default=100, help='elimination batch size for "bico_elim"')
    parser.add_argument('--exch-steps', type=int, default=20,
                        help='exchange rounds for "bico_exch" (paper: 200; see module docstring)')
    parser.add_argument('--reg-outer-it', type=int, default=150,
                        help='outer iterations for "bico_reg" (Algorithm 2); raise this first if the '
                             'BiCo Reg curve looks noisy/non-monotonic -- small target sizes need many '
                             'more beta-doublings to converge within the budget than large ones. The '
                             'paper does not report a fixed per-size outer-iteration count for BiCo Reg '
                             'in Sec. 5.1 -- it uses a fixed *total* outer-step budget instead (this is '
                             'also why the paper\'s own Fig. 3 is non-monotonic between the 20%% and 90%% '
                             'points); 150 here is just a practical starting default, not a paper value')
    parser.add_argument('--reg-warm-inner-it', type=int, default=30,
                        help='inner GD steps to re-fit the ConvNet after each weight update in "bico_reg" '
                             '(raise this together with --reg-outer-it if the curve is noisy)')
    parser.add_argument('--reg-patience', type=int, default=3,
                        help='plateau length (in outer iterations) before beta is doubled in "bico_reg"')
    parser.add_argument('--reg-beta', type=float, default=1e-7,
                        help='initial sparsity penalty for "bico_reg" (paper: 1e-7, Appendix C)')
    parser.add_argument('--inner-reg', type=float, default=1e-7,
                        help='lambda of the inner ridge penalty, Eq. (8) (paper: 1e-7, Appendix C, for '
                             'all Sec. 3.5 variants). The paper derives this under a convex inner problem; '
                             'a ConvNet is non-convex, so this may need retuning -- raise it if inner '
                             'optimization looks unstable')
    parser.add_argument('--retrain-from-scratch', action='store_true',
                        help='fully retrain the ConvNet from scratch after every selection/weight-update '
                             'step, for *all* methods (bico_fwd/fwd_b/elim/exch/reg), instead of '
                             'warm-starting. Paper\'s Appendix C ("Variants") warm-starts every Sec. 3.5 '
                             'method by default; retraining from scratch is only used for the Sec. 5.2.3 '
                             'deep-net experiments (WideResNet with a cosine-annealed LR schedule). Try '
                             'this if warm-starting looks unstable for a given method on the real ConvNet')
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
