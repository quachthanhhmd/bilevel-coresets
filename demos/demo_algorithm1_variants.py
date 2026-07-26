"""Practical variants of Algorithm 1 -- Sec. 5.1 / Figure 3 of the JMLR paper.

Reproduces the comparison of Figure 3: *test accuracy vs. subset size* for
multiclass logistic regression, comparing the practical approximations of
Sec. 3.5 to plain uniform sampling and to training on the full data set:

    * ``Uniform``    -- uniform random subsampling,
    * ``BiCo Fwd``   -- one-by-one forward selection, binary weights
      (``selection_batch_size`` kept small),
    * ``BiCo Fwd b`` -- forward selection in larger batches (Sec. 3.5.1) --
      the paper shows this incurs an early performance penalty that
      disappears once ~25% of the points are selected,
    * ``BiCo Elim``  -- elimination in batches: start from the full data set
      and repeatedly drop the least useful points,
    * ``BiCo Exch``  -- exchange (Fedorov-style "excursion"): repeatedly swap
      the worst selected points for the best candidates,
    * ``BiCo Reg``   -- the weighted coreset of Algorithm 2 (Sec. 3.5.3,
      ``bicoreset.regularized.RegularizedBilevelCoreset``); the paper shows
      this is by far the most compact (matches full-data accuracy at ~8% of
      the points).

The paper's own setup targets q=2048-dimensional Nystrom-CNTK features of
CIFAR-10 (a Convolutional Neural Tangent Kernel proxy, Sec. 5.1), which
needs jax/neural-tangents and heavy compute unavailable in this sandbox. We
substitute a real, offline, *multiclass* problem of comparable spirit that
already ships with scikit-learn: raw 64-dimensional pixel features of the
``digits`` data set (10 classes, 1797 points) with the same target model
(multiclass logistic regression). The qualitative comparison between the
Algorithm 1 variants is what this demo reproduces; the exact percentages
will differ from Figure 3 because the feature space/model capacity differ.

The defaults below are the paper's literal Sec. 5.1 hyperparameters: one-by-one
(``b=1``) forward selection for "BiCo Fwd", batches of 25 for "BiCo Fwd b",
elimination in batches of 200, exchange with 200 steps (1% of the selected
points exchanged per step), and 100 conjugate-gradient steps per implicit
gradient. This is slow on CPU for the larger subset sizes/candidate pools --
pass ``--fast`` for a scaled-down, CPU-friendly approximation instead (useful
for a quick smoke test; the numbers will then no longer match the paper's
setup).

Run with::

    python demos/demo_algorithm1_variants.py
    python demos/demo_algorithm1_variants.py --fast   # quick CPU smoke test
"""

import argparse
import time

import numpy as np
import torch

from _common import set_seed, train_test_split, evaluate_subset

from bicoreset import losses
from bicoreset.direct import BilevelCoreset
from bicoreset.regularized import RegularizedBilevelCoreset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHODS = ['Uniform', 'BiCo Fwd', 'BiCo Fwd b', 'BiCo Elim', 'BiCo Exch', 'BiCo Reg']
COLORS = {'Uniform': '#2a78d6', 'BiCo Fwd': '#008300', 'BiCo Fwd b': '#5aa95a',
          'BiCo Elim': '#e34948', 'BiCo Exch': '#e58a1a', 'BiCo Reg': '#4a3aa7'}
MARKERS = {'Uniform': 'o', 'BiCo Fwd': 'D', 'BiCo Fwd b': 's',
           'BiCo Elim': 'v', 'BiCo Exch': '^', 'BiCo Reg': 'P'}


def load_dataset():
    """Real, offline multiclass problem substituting for CIFAR-10/CNTK (see docstring)."""
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler

    d = load_digits()
    X = StandardScaler().fit_transform(d.data)
    return (torch.from_numpy(X).float(), torch.from_numpy(d.target).long(),
           'digits (64d, 10 classes, {}pts)'.format(X.shape[0]))


def fwd_builder(dim, n_classes, args):
    import models
    return BilevelCoreset(
        model_fn=lambda: models.LogisticRegression(dim, n_classes),
        loss_fn=losses.cross_entropy,
        inner_reg=1e-3,
        ihvp='cg', ihvp_kwargs={'max_iter': args.cg_iters, 'damping': 1e-3},
        max_inner_it=args.max_inner_it, inner_lr=0.05,
        max_outer_it=0,  # binary weights, Sec. 3.5.1
        candidate_pool_size=args.candidate_pool_size,
        retrain_from_scratch=True,
        verbose=False)


def reg_builder(dim, n_classes, args):
    import models
    return RegularizedBilevelCoreset(
        model_fn=lambda: models.LogisticRegression(dim, n_classes),
        loss_fn=losses.cross_entropy,
        inner_reg=1e-3,
        beta=1e-6, adaptive_beta=True, patience=3,
        max_outer_it=args.reg_outer_it,
        outer_lr=0.05,
        max_inner_it=args.max_inner_it, warm_inner_it=15,
        ihvp='cg', ihvp_kwargs={'max_iter': args.cg_iters, 'damping': 1e-3},
        retrain_from_scratch=False,
        logging_period=10_000,
        verbose=False)


def run_size(X_tr, y_tr, X_te, y_te, m, n_classes, seed, args):
    dim = X_tr.shape[1]
    n = X_tr.shape[0]
    rs = np.random.RandomState(seed)
    out = {}
    times = {}

    out['Uniform'] = evaluate_subset(X_tr, y_tr, X_te, y_te,
                                     rs.choice(n, m, replace=False), n_classes)

    start = min(10, m)
    # Paper (Sec. 5.1): one-by-one forward selection (b=1) vs. forward
    # selection in batches of 25; elimination in batches of 200; exchange
    # with 200 steps, each exchanging 1% of the selected points. --fast
    # scales these down for a quick CPU smoke test (see module docstring).
    if args.fast:
        b_fine = max(1, m // 40)
        b_coarse = max(1, m // 8)
        elim_batch = max(1, (n - m) // args.elim_steps)
        exch_batch = max(1, m // 10)
        exch_steps = args.exch_steps
    else:
        b_fine = 1
        b_coarse = args.fwd_b_batch
        elim_batch = args.elim_batch
        exch_batch = max(1, int(round(0.01 * m)))
        exch_steps = args.exch_steps

    t0 = time.time()
    builder = fwd_builder(dim, n_classes, args)
    inds, w = builder.build(X_tr, y_tr, m, strategy='forward',
                            selection_batch_size=b_fine, start_size=start)
    times['BiCo Fwd'] = time.time() - t0
    out['BiCo Fwd'] = evaluate_subset(X_tr, y_tr, X_te, y_te, inds, n_classes)

    t0 = time.time()
    builder = fwd_builder(dim, n_classes, args)
    inds, w = builder.build(X_tr, y_tr, m, strategy='forward',
                            selection_batch_size=b_coarse, start_size=start)
    times['BiCo Fwd b'] = time.time() - t0
    out['BiCo Fwd b'] = evaluate_subset(X_tr, y_tr, X_te, y_te, inds, n_classes)

    t0 = time.time()
    builder = fwd_builder(dim, n_classes, args)
    inds, w = builder.build(X_tr, y_tr, m, strategy='elimination',
                            selection_batch_size=elim_batch)
    times['BiCo Elim'] = time.time() - t0
    out['BiCo Elim'] = evaluate_subset(X_tr, y_tr, X_te, y_te, inds, n_classes)

    t0 = time.time()
    builder = fwd_builder(dim, n_classes, args)
    inds, w = builder.build(X_tr, y_tr, m, strategy='exchange',
                            selection_batch_size=exch_batch,
                            n_exchange_steps=exch_steps)
    times['BiCo Exch'] = time.time() - t0
    out['BiCo Exch'] = evaluate_subset(X_tr, y_tr, X_te, y_te, inds, n_classes)

    t0 = time.time()
    builder = reg_builder(dim, n_classes, args)
    inds, w = builder.build(X_tr, y_tr, m=m)
    times['BiCo Reg'] = time.time() - t0
    out['BiCo Reg'] = evaluate_subset(X_tr, y_tr, X_te, y_te, inds, n_classes, weights=w)

    return out, times


def main():
    args = parse_args()
    fractions = [float(s) for s in args.fractions.split(',')]

    X, y, name = load_dataset()
    n_classes = int(y.max().item()) + 1

    accs = {method: [[] for _ in fractions] for method in METHODS}
    full_accs = []

    for seed in range(args.seeds):
        set_seed(seed)
        X_tr, y_tr, X_te, y_te = train_test_split(X, y, seed=seed)
        full_accs.append(evaluate_subset(X_tr, y_tr, X_te, y_te, np.arange(len(y_tr)), n_classes))
    full_acc = float(np.mean(full_accs))
    n_train = int(len(y) * 0.7)
    print('== {} == (full-data test accuracy = {:.4f}, n_train={}) =='.format(name, full_acc, n_train))

    for fi, frac in enumerate(fractions):
        m = max(10, int(round(frac * n_train)))
        agg_times = {method: [] for method in METHODS[1:]}
        for seed in range(args.seeds):
            set_seed(seed)
            X_tr, y_tr, X_te, y_te = train_test_split(X, y, seed=seed)
            out, times = run_size(X_tr, y_tr, X_te, y_te, m, n_classes, seed, args)
            for method in METHODS:
                accs[method][fi].append(out[method])
            for method, t in times.items():
                agg_times[method].append(t)
        line = 'frac={:>5.1%} (m={:4d})  '.format(frac, m) + '  '.join(
            '{}={:.4f}'.format(meth, np.mean(accs[meth][fi])) for meth in METHODS)
        print(line)
        print('   construction time (s): ' + '  '.join(
            '{}={:.2f}'.format(meth, np.mean(t)) for meth, t in agg_times.items()))

    import os
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'algorithm1_variants_accuracy.png')
    plot_results(name, fractions, accs, full_acc, output_path)


def plot_results(name, fractions, accs, full_acc, output_path):
    import seaborn as sns
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 11})

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = [f * 100 for f in fractions]
    for method in METHODS:
        means = [np.mean(accs[method][fi]) for fi in range(len(fractions))]
        stds = [np.std(accs[method][fi]) for fi in range(len(fractions))]
        ax.errorbar(x, means, yerr=stds, label=method, color=COLORS[method],
                   marker=MARKERS[method], markersize=8, markeredgecolor='black',
                   capsize=3, linewidth=1.7)
    ax.axhline(full_acc, color='black', linestyle='--', linewidth=1.4, label='Full Dataset')
    ax.set_xscale('log')
    ax.set_xlabel('Subset Size (% of training set)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Practical variants of Algorithm 1\n{}'.format(name))
    ax.legend(frameon=True, fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print('\nSaved plot to {}'.format(output_path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fractions', default='0.01,0.03,0.08,0.2,0.4,0.6',
                        help='comma-separated subset sizes, as a fraction of the training set')
    parser.add_argument('--seeds', type=int, default=1, help='number of random seeds to average over')
    parser.add_argument('--max-inner-it', type=int, default=100,
                        help='inner GD steps to fit logistic regression on the current support')
    parser.add_argument('--cg-iters', type=int, default=100,
                        help='conjugate-gradient steps for the implicit gradient (paper: 100)')
    parser.add_argument('--candidate-pool-size', type=int, default=None,
                        help='random candidate subsample scored per forward-selection step '
                             '(paper: none, scores all remaining candidates every step)')
    parser.add_argument('--fwd-b-batch', type=int, default=25,
                        help='batch size for "BiCo Fwd b" (paper: 25)')
    parser.add_argument('--elim-batch', type=int, default=200,
                        help='elimination batch size for "BiCo Elim" (paper: 200)')
    parser.add_argument('--elim-steps', type=int, default=10,
                        help='number of elimination rounds, only used with --fast')
    parser.add_argument('--exch-steps', type=int, default=200,
                        help='number of exchange rounds (paper: 200; each step exchanges '
                             '1%% of the selected points unless --fast is set)')
    parser.add_argument('--reg-outer-it', type=int, default=60,
                        help='outer iterations for BiCo Reg (Algorithm 2); more -> smaller/better support')
    parser.add_argument('--fast', action='store_true',
                        help='scale down batch sizes/step counts for a quick CPU smoke test '
                             '(numbers will then no longer match the paper; see module docstring)')
    parser.add_argument('--output', default=None, help='output PNG path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
