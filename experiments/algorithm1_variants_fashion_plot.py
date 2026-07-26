"""Plot the Sec. 5.1 / Figure 3 comparison from real FashionMNIST results.

Reads the JSON result files produced by ``run_algorithm1_variants_fashion.py``
(methods: uniform, bico_fwd, bico_fwd_b, bico_elim, bico_exch, bico_reg, plus
the ``full`` reference) from ``--results-dir`` and plots test accuracy vs.
coreset size, in the same style as Figure 3 of the JMLR paper / as
``demos/demo_algorithm1_variants.py``.

Unlike ``baseline_comparison_plot.py`` this script does *not* fabricate
synthetic fallback numbers for missing (method, size, seed) combinations --
there is no established reference result for this experiment yet. Points
with no matching result file are simply skipped (and reported), so run
``run_algorithm1_variants_fashion.py`` for every method/size/seed you want to
appear before plotting.

Usage:
    python experiments/algorithm1_variants_fashion_plot.py --sizes 50,100,200,400,800 --seeds 0,1,2
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

METHODS = {
    'Uniform': 'uniform',
    'BiCo Fwd': 'bico_fwd',
    'BiCo Fwd b': 'bico_fwd_b',
    'BiCo Elim': 'bico_elim',
    'BiCo Exch': 'bico_exch',
    'BiCo Reg': 'bico_reg',
}
COLORS = {'Uniform': '#2a78d6', 'BiCo Fwd': '#008300', 'BiCo Fwd b': '#5aa95a',
          'BiCo Elim': '#e34948', 'BiCo Exch': '#e58a1a', 'BiCo Reg': '#4a3aa7'}
MARKERS = {'Uniform': 'o', 'BiCo Fwd': 'D', 'BiCo Fwd b': 's',
           'BiCo Elim': 'v', 'BiCo Exch': '^', 'BiCo Reg': 'P'}


def get_results_dir(args):
    if args.results_dir:
        return args.results_dir
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'algo1_fashion_results')


def load_point(results_dir, method, size, seed):
    path = os.path.join(results_dir, '{}_{}_{}.txt'.format(method, size, seed))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


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
    sizes = [int(s) for s in args.sizes.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    results_dir = get_results_dir(args)

    accs = {name: [[] for _ in sizes] for name in METHODS}
    missing = []
    for display_name, method in METHODS.items():
        for si, size in enumerate(sizes):
            for seed in seeds:
                point = load_point(results_dir, method, size, seed)
                if point is None:
                    missing.append('{}_{}_{}'.format(method, size, seed))
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
        for si, size in enumerate(sizes):
            if accs[display_name][si]:
                xs.append(size)
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

    ax.set_xscale('log')
    ax.set_xlabel('Coreset size (number of images)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Practical variants of Algorithm 1 on FashionMNIST')
    ax.legend(frameon=True, fontsize=9, loc='lower right')
    if missing:
        fig.text(0.01, 0.005, '* {} (method,size,seed) combinations have no result file yet'.format(len(missing)),
                fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'algorithm1_variants_fashion_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print('Saved plot to {}'.format(output_path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='50,100,200,400,800',
                        help='comma-separated coreset sizes to look for result files at')
    parser.add_argument('--seeds', default='0,1,2', help='comma-separated seeds to average over')
    parser.add_argument('--results-dir', default=None)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    main()
