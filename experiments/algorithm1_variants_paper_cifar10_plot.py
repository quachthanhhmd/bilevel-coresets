"""Plot the Sec. 5.1 / Figure 3 reproduction from real CIFAR-10 (or the same
CNTK-Nystrom-proxy pipeline applied to FashionMNIST) results.

Reads the JSON result files produced by
``run_algorithm1_variants_paper_cifar10.py`` (methods: uniform, bico_fwd,
bico_fwd25, bico_elim, bico_exch, bico_reg, plus the ``full`` reference) from
``--results-dir`` and plots test accuracy vs. coreset size (as a %% of the
train partition), in the same style as Figure 3 of the JMLR paper.

Pass ``--dataset fashionmnist`` to plot the FashionMNIST run instead of the
default ``cifar10`` (matches ``run_algorithm1_variants_paper_cifar10.py``'s
default output directory naming, ``algo1_paper_<dataset>_results``). Note the
two are testing different things (paper reproduction vs. transfer of the same
methodology to a different dataset) -- don't plot them on the same axes as if
comparing like-for-like without saying so.

Like ``algorithm1_variants_fashion_plot.py``, this does *not* fabricate
synthetic fallback numbers for missing (method, size, seed) combinations --
points with no matching result file are simply skipped (and reported).

Usage:
    python experiments/algorithm1_variants_paper_cifar10_plot.py --sizes-pct 0.5,2,8,32 --seeds 0,1,2
    python experiments/algorithm1_variants_paper_cifar10_plot.py --dataset fashionmnist --sizes-pct 0.5,2,8,32 --seeds 0
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns

METHODS = {
    'Uniform': 'uniform',
    'BiCo Fwd': 'bico_fwd',
    'BiCo Fwd 25': 'bico_fwd25',
    'BiCo Elim': 'bico_elim',
    'BiCo Exch': 'bico_exch',
    'BiCo Reg': 'bico_reg',
}
COLORS = {'Uniform': '#2a78d6', 'BiCo Fwd': '#008300', 'BiCo Fwd 25': '#5aa95a',
          'BiCo Elim': '#e34948', 'BiCo Exch': '#e58a1a', 'BiCo Reg': '#4a3aa7'}
MARKERS = {'Uniform': 'o', 'BiCo Fwd': 'D', 'BiCo Fwd 25': 's',
           'BiCo Elim': 'v', 'BiCo Exch': '^', 'BiCo Reg': 'P'}


def get_results_dir(args):
    if args.results_dir:
        return args.results_dir
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'algo1_paper_{}_results'.format(args.dataset))


def load_point(results_dir, method, size_pct, seed):
    """Result files are named ``{method}_{absolute_size}_{seed}.txt`` (the
    absolute size depends on the train-pool size used at run time), so we
    match on ``size_pct`` stored *inside* the JSON instead of guessing the
    absolute count from the filename."""
    pattern = os.path.join(results_dir, '{}_*_{}.txt'.format(method, seed))
    for path in glob.glob(pattern):
        with open(path) as f:
            data = json.load(f)
        if abs(data.get('size_pct', -1) - size_pct) < 1e-6:
            return data
    return None


def load_full(results_dir, seeds):
    accs = []
    for seed in seeds:
        path = os.path.join(results_dir, 'full_{}.txt'.format(seed))
        if os.path.exists(path):
            with open(path) as f:
                accs.append(json.load(f)['test_acc'])
    return accs


def main():
    args = parse_args()
    sizes_pct = [float(s) for s in args.sizes_pct.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    results_dir = get_results_dir(args)

    accs = {name: [[] for _ in sizes_pct] for name in METHODS}
    missing = []
    for display_name, method in METHODS.items():
        for si, size_pct in enumerate(sizes_pct):
            for seed in seeds:
                point = load_point(results_dir, method, size_pct, seed)
                if point is None:
                    missing.append('{}_{}%_{}'.format(method, size_pct, seed))
                    continue
                accs[display_name][si].append(point['test_acc'])

    full_accs = load_full(results_dir, seeds)
    if not full_accs:
        print('Warning: no full_<seed>.txt reference file found in {}; '
              'run with --method full first for a dashed reference line.'.format(results_dir))

    if missing:
        print('Missing result files (skipped in the plot):')
        for m in missing:
            print('  ' + m)

    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for display_name in METHODS:
        xs, means, stds = [], [], []
        for si, size_pct in enumerate(sizes_pct):
            if accs[display_name][si]:
                xs.append(size_pct)
                means.append(float(np.mean(accs[display_name][si])))
                stds.append(float(np.std(accs[display_name][si])))
        if not xs:
            continue
        ax.errorbar(xs, means, yerr=stds, label=display_name, color=COLORS[display_name],
                   marker=MARKERS[display_name], markersize=8, markeredgecolor='black',
                   capsize=3, linewidth=1.7)

    if full_accs:
        ax.axhline(float(np.mean(full_accs)), color='black', linestyle='--',
                  linewidth=1.4, label='Full Dataset')

    dataset_label = {'cifar10': 'CIFAR-10', 'fashionmnist': 'FashionMNIST'}[args.dataset]
    title = 'Bilevel coresets for logistic regression on CNTK-Nystrom features of {}'.format(dataset_label)
    if args.dataset != 'cifar10':
        title += ' (not Fig. 3 -- same pipeline, different dataset)'

    ax.set_xscale('log')
    # same fix as algorithm1_variants_fashion_plot.py: force a labeled tick at
    # every size_pct actually used instead of relying on the default log
    # locator, which only labels round powers of 10 in range.
    ax.set_xticks(sizes_pct)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Subset Size (% of the train partition)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=9, loc='lower right')
    if missing:
        fig.text(0.01, 0.005, '* {} (method,size,seed) combinations have no result file yet'.format(len(missing)),
                fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'algorithm1_variants_paper_{}_accuracy.png'.format(args.dataset))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print('Saved plot to {}'.format(output_path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10', 'fashionmnist'], default='cifar10')
    parser.add_argument('--sizes-pct', default='0.5,2,8,32',
                        help='comma-separated coreset sizes (as %% of the train partition) to look for')
    parser.add_argument('--seeds', default='0', help='comma-separated seeds to average over')
    parser.add_argument('--results-dir', default=None)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    main()
