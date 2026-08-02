"""Binary logistic regression coresets -- Sec. 5.2.2 / Figure 6 of the JMLR paper.

Reproduces the comparison of Figure 6: *test accuracy vs. coreset size* for
binary logistic regression, comparing

    * ``BiCo w/ Weights`` -- Algorithm 1, one-by-one forward selection with
      the weight optimization of line 6 (paper: 150 outer iterations,
      reduced here via ``--outer-iters`` for a CPU-friendly demo),
    * ``BiCo``            -- the unweighted (binary-weight) variant, much
      cheaper to construct since it skips the weight optimization loop
      entirely (the paper reports a 150x construction speedup, matching the
      number of outer iterations saved),
    * ``k-means++``        -- Arthur & Vassilvitskii (2007),
    * ``Sensitivity Coreset`` -- Huggins et al. (2016),

against the accuracy obtained by training on the *full* data set (dashed
line). This mirrors the paper's own baselines for this section (it also
tried Hilbert coresets but dropped them for underperforming uniform
sampling, so we don't include them either).

The paper uses four LIBSVM binary classification data sets fetched over the
network (9k-600k points, 8-123 features). This environment has no internet
access, so by default we substitute two *real*, offline data sets already
bundled with scikit-learn, of comparable spirit: ``breast_cancer`` (30
features, 569 points) and ``digits`` restricted to the visually-similar
classes ``3`` vs ``8`` (64 features, ~360 points).

Pass ``--dataset fashion`` to instead run on real FashionMNIST, restricted to
two commonly-confused classes (default: Pullover vs Shirt, the hardest pair
in FashionMNIST) flattened to 784-d pixel vectors -- this needs torchvision
+ internet (e.g. on Kaggle), unavailable in this sandbox. ``--dataset all``
runs both the sklearn pair and FashionMNIST (on the *same* combined figure).

Pass ``--dataset cifar10`` for real CIFAR-10 instead, restricted to two
commonly-confused classes (default: Cat vs Dog, a standard "hardest pair" for
CIFAR-10 in the literature) flattened to 3072-d pixel vectors -- same
torchvision/internet requirement as ``fashion``. This is *not* included in
``--dataset all`` on purpose: run ``--dataset fashion`` and ``--dataset
cifar10`` as two separate invocations (with different ``--output`` paths) to
keep the two on separate figures rather than combined into one, since they
answer separate "does the paper's Sec. 5.2.2 setup transfer to dataset X"
questions rather than a single comparison.

Pass ``--dataset kmnist`` for real KMNIST (Kuzushiji-MNIST), same treatment
as ``fashion``/``cifar10``: two classes flattened to 784-d pixel vectors,
same torchvision/internet requirement, also kept off ``--dataset all`` and
run as its own separate invocation/figure. Unlike ``fashion``'s
Pullover-vs-Shirt or ``cifar10``'s Cat-vs-Dog, there is no well-known
"hardest confused pair" reference for KMNIST readily available offline, so
the default classes (0, 1) are an arbitrary (not empirically-chosen)
placeholder -- pass ``--kmnist-classes`` to pick a different pair if you
have a specific comparison in mind.

Run with::

    python demos/demo_logistic_regression.py
    python demos/demo_logistic_regression.py --sizes 5,15,30,50 --seeds 3 --outer-iters 20
    python demos/demo_logistic_regression.py --dataset fashion --sizes 10,30,60,100 \\
        --output demos/logistic_regression_coreset_accuracy_fashion.png
    python demos/demo_logistic_regression.py --dataset cifar10 --sizes 10,30,60,100 \\
        --output demos/logistic_regression_coreset_accuracy_cifar10.png
    python demos/demo_logistic_regression.py --dataset kmnist --sizes 10,30,60,100 \\
        --output demos/logistic_regression_coreset_accuracy_kmnist.png
"""

import argparse
import time

import numpy as np
import torch

from _common import set_seed, train_test_split, evaluate_subset

from bicoreset import losses
from bicoreset.direct import BilevelCoreset
from cl_streaming import summary

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHODS = ['BiCo w/ Weights', 'BiCo', 'k-means++', 'Sensitivity Coreset']
COLORS = {'BiCo w/ Weights': '#008300', 'BiCo': '#5aa95a',
          'k-means++': '#e58a1a', 'Sensitivity Coreset': '#4a3aa7'}
MARKERS = {'BiCo w/ Weights': 'D', 'BiCo': 'o', 'k-means++': 'v', 'Sensitivity Coreset': '^'}


def load_sklearn_datasets():
    """Two real, offline binary classification problems (see module docstring)."""
    from sklearn.datasets import load_breast_cancer, load_digits
    from sklearn.preprocessing import StandardScaler

    datasets = {}

    d = load_breast_cancer()
    X = StandardScaler().fit_transform(d.data)
    datasets['breast_cancer (30d, {}pts)'.format(X.shape[0])] = (
        torch.from_numpy(X).float(), torch.from_numpy(d.target).long())

    digits = load_digits()
    mask = np.isin(digits.target, [3, 8])
    X = StandardScaler().fit_transform(digits.data[mask])
    y = (digits.target[mask] == 8).astype(np.int64)
    datasets['digits 3-vs-8 (64d, {}pts)'.format(int(mask.sum()))] = (
        torch.from_numpy(X).float(), torch.from_numpy(y).long())

    return datasets


def load_fashion_dataset(classes=(2, 6), max_samples=4000, data_root='data', seed=0):
    """Real FashionMNIST restricted to two classes, flattened to 784-d pixels.

    Default classes (2, 6) = Pullover vs Shirt, the single hardest pair in
    FashionMNIST (the two classes most often confused by a CNN classifier),
    chosen for the same reason ``digits`` uses ``3`` vs ``8`` -- a real,
    non-trivial binary problem, not an easy one. Needs torchvision + internet
    (e.g. on Kaggle); not available in this sandbox.
    """
    import torchvision.datasets as tv_datasets
    import torchvision.transforms as transforms
    from sklearn.preprocessing import StandardScaler

    transform = transforms.Compose([transforms.ToTensor()])
    train = tv_datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test = tv_datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

    imgs, labels = [], []
    for ds in (train, test):
        targets = np.asarray(ds.targets)
        mask = np.isin(targets, classes)
        idx = np.nonzero(mask)[0]
        for i in idx:
            img, label = ds[i]
            imgs.append(img.view(-1).numpy())
            labels.append(label)
    X = np.stack(imgs).astype(np.float32)
    y = (np.asarray(labels) == classes[1]).astype(np.int64)

    if max_samples is not None and X.shape[0] > max_samples:
        rs = np.random.RandomState(seed)
        keep = rs.choice(X.shape[0], max_samples, replace=False)
        X, y = X[keep], y[keep]

    X = StandardScaler().fit_transform(X)
    class_names = {2: 'Pullover', 6: 'Shirt', 0: 'T-shirt/top', 4: 'Coat'}
    a = class_names.get(classes[0], str(classes[0]))
    b = class_names.get(classes[1], str(classes[1]))
    name = 'fashion {}-vs-{}'.format(a, b, X.shape[0])
    return name, torch.from_numpy(X).float(), torch.from_numpy(y).long()


def load_cifar10_dataset(classes=(3, 5), max_samples=4000, data_root='data', seed=0):
    """Real CIFAR-10 restricted to two classes, flattened to 3072-d pixels.

    Default classes (3, 5) = Cat vs Dog, a standard "hardest pair" for
    CIFAR-10 (visually similar, most-confused pair for CNN classifiers in the
    literature) -- same reasoning as ``fashion``'s Pullover-vs-Shirt default.
    Needs torchvision + internet (e.g. on Kaggle); not available in this
    sandbox.
    """
    import torchvision.datasets as tv_datasets
    import torchvision.transforms as transforms
    from sklearn.preprocessing import StandardScaler

    transform = transforms.Compose([transforms.ToTensor()])
    train = tv_datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test = tv_datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    imgs, labels = [], []
    for ds in (train, test):
        targets = np.asarray(ds.targets)
        mask = np.isin(targets, classes)
        idx = np.nonzero(mask)[0]
        for i in idx:
            img, label = ds[i]
            imgs.append(img.view(-1).numpy())
            labels.append(label)
    X = np.stack(imgs).astype(np.float32)
    y = (np.asarray(labels) == classes[1]).astype(np.int64)

    if max_samples is not None and X.shape[0] > max_samples:
        rs = np.random.RandomState(seed)
        keep = rs.choice(X.shape[0], max_samples, replace=False)
        X, y = X[keep], y[keep]

    X = StandardScaler().fit_transform(X)
    class_names = {0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer',
                   5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'}
    a = class_names.get(classes[0], str(classes[0]))
    b = class_names.get(classes[1], str(classes[1]))
    name = 'cifar10 {}-vs-{}'.format(a, b, X.shape[0])
    return name, torch.from_numpy(X).float(), torch.from_numpy(y).long()


def load_kmnist_dataset(classes=(0, 1), max_samples=4000, data_root='data', seed=0):
    """Real KMNIST (Kuzushiji-MNIST) restricted to two classes, flattened to
    784-d pixels. Same shape/format as FashionMNIST, so this mirrors
    :func:`load_fashion_dataset` exactly. Needs torchvision + internet (e.g.
    on Kaggle); not available in this sandbox. Default classes are an
    arbitrary placeholder pair -- see module docstring."""
    import torchvision.datasets as tv_datasets
    import torchvision.transforms as transforms
    from sklearn.preprocessing import StandardScaler

    transform = transforms.Compose([transforms.ToTensor()])
    train = tv_datasets.KMNIST(root=data_root, train=True, download=True, transform=transform)
    test = tv_datasets.KMNIST(root=data_root, train=False, download=True, transform=transform)

    imgs, labels = [], []
    for ds in (train, test):
        targets = np.asarray(ds.targets)
        mask = np.isin(targets, classes)
        idx = np.nonzero(mask)[0]
        for i in idx:
            img, label = ds[i]
            imgs.append(img.view(-1).numpy())
            labels.append(label)
    X = np.stack(imgs).astype(np.float32)
    y = (np.asarray(labels) == classes[1]).astype(np.int64)

    if max_samples is not None and X.shape[0] > max_samples:
        rs = np.random.RandomState(seed)
        keep = rs.choice(X.shape[0], max_samples, replace=False)
        X, y = X[keep], y[keep]

    X = StandardScaler().fit_transform(X)
    class_names = {0: 'o', 1: 'ki', 2: 'su', 3: 'tsu', 4: 'na', 5: 'ha', 6: 'ma', 7: 'ya', 8: 're', 9: 'wo'}
    a = class_names.get(classes[0], str(classes[0]))
    b = class_names.get(classes[1], str(classes[1]))
    name = 'kmnist {}-vs-{}'.format(a, b, X.shape[0])
    return name, torch.from_numpy(X).float(), torch.from_numpy(y).long()


def load_datasets(which='sklearn', fashion_classes=(2, 6), fashion_max_samples=4000,
                  cifar10_classes=(3, 5), cifar10_max_samples=4000,
                  kmnist_classes=(0, 1), kmnist_max_samples=4000):
    datasets = {}
    if which in ('sklearn', 'all'):
        datasets.update(load_sklearn_datasets())
    if which in ('fashion', 'all'):
        name, X, y = load_fashion_dataset(classes=fashion_classes, max_samples=fashion_max_samples)
        datasets[name] = (X, y)
    if which == 'cifar10':
        # deliberately NOT included in 'all' -- kept on its own figure, see
        # module docstring (don't combine with fashion/sklearn on one chart)
        name, X, y = load_cifar10_dataset(classes=cifar10_classes, max_samples=cifar10_max_samples)
        datasets[name] = (X, y)
    if which == 'kmnist':
        # also deliberately NOT included in 'all', same reasoning as cifar10
        name, X, y = load_kmnist_dataset(classes=kmnist_classes, max_samples=kmnist_max_samples)
        datasets[name] = (X, y)
    return datasets


def make_builder(dim, weighted, args):
    import models
    return BilevelCoreset(
        model_fn=lambda: models.LogisticRegression(dim, 2),
        loss_fn=losses.cross_entropy,
        inner_reg=1e-2,  # paper: L2-penalty 0.01 for the logistic regression
        ihvp='cg', ihvp_kwargs={'max_iter': args.cg_iters, 'damping': 1e-3},
        max_inner_it=args.max_inner_it, inner_lr=0.05,
        max_outer_it=args.outer_iters if weighted else 0,
        outer_lr=0.01, warm_inner_it=args.warm_inner_it,
        candidate_pool_size=None,
        retrain_from_scratch=True,
        verbose=False)


def run_dataset(name, X, y, sizes, n_seeds, args):
    dim = X.shape[1]
    accs = {m: [[] for _ in sizes] for m in METHODS}
    time_weighted, time_unweighted = [], []

    full_accs = []
    for seed in range(n_seeds):
        set_seed(seed)
        X_tr, y_tr, X_te, y_te = train_test_split(X, y, seed=seed)
        full_accs.append(evaluate_subset(X_tr, y_tr, X_te, y_te, np.arange(len(y_tr)), 2))
    full_acc = float(np.mean(full_accs))
    print('\n== {} == (full-data test accuracy = {:.4f}) =='.format(name, full_acc))

    for si, m in enumerate(sizes):
        for seed in range(n_seeds):
            set_seed(seed)
            X_tr, y_tr, X_te, y_te = train_test_split(X, y, seed=seed)
            X_np, y_np = X_tr.numpy(), y_tr.numpy()
            rs = np.random.RandomState(seed)
            start = min(10, m)
            # Paper: strictly one-by-one forward selection (b=1). We batch b>1 for
            # larger m purely to keep the CPU demo fast (Sec. 3.5.1 batching, Fig. 3);
            # pass --batch-size 1 for a faithful (but much slower) reproduction.
            default_b = 1 if m <= 10 else max(1, m // 10)
            b = max(1, min(args.batch_size or default_b, m - start))

            t0 = time.time()
            builder = make_builder(dim, weighted=True, args=args)
            inds, w = builder.build(X_tr, y_tr, m, strategy='forward',
                                    selection_batch_size=b, start_size=start)
            time_weighted.append(time.time() - t0)
            accs['BiCo w/ Weights'][si].append(
                evaluate_subset(X_tr, y_tr, X_te, y_te, inds, 2, weights=w))

            t0 = time.time()
            builder = make_builder(dim, weighted=False, args=args)
            inds, w = builder.build(X_tr, y_tr, m, strategy='forward',
                                    selection_batch_size=b, start_size=start)
            time_unweighted.append(time.time() - t0)
            accs['BiCo'][si].append(evaluate_subset(X_tr, y_tr, X_te, y_te, inds, 2))

            inds = summary.Summarizer.factory('kmeans_features', rs).build_summary(X_np, y_np, m)
            accs['k-means++'][si].append(evaluate_subset(X_tr, y_tr, X_te, y_te, inds, 2))

            inds = summary.Summarizer.factory('sensitivity', rs).build_summary(X_np, y_np, m)
            accs['Sensitivity Coreset'][si].append(evaluate_subset(X_tr, y_tr, X_te, y_te, inds, 2))

        line = 'size={:3d}  '.format(m) + '  '.join(
            '{}={:.4f}'.format(meth, np.mean(accs[meth][si])) for meth in METHODS)
        print(line)

    speedup = np.mean(time_weighted) / max(1e-9, np.mean(time_unweighted))
    print('avg construction time: BiCo w/ Weights={:.2f}s, BiCo={:.2f}s (weighted is {:.1f}x slower)'.format(
        np.mean(time_weighted), np.mean(time_unweighted), speedup))

    return accs, full_acc


def plot_results(all_results, sizes, output_path):
    import seaborn as sns
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 11})

    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.8))
    if n == 1:
        axes = [axes]

    for ax, (name, (accs, full_acc)) in zip(axes, all_results.items()):
        for method in METHODS:
            means = [np.mean(accs[method][si]) for si in range(len(sizes))]
            stds = [np.std(accs[method][si]) for si in range(len(sizes))]
            ax.errorbar(sizes, means, yerr=stds, label=method, color=COLORS[method],
                       marker=MARKERS[method], markersize=7, markeredgecolor='black',
                       capsize=3, linewidth=1.6)
        ax.axhline(full_acc, color='black', linestyle='--', linewidth=1.2, label='Full data set')
        ax.set_xlabel('Coreset size')
        ax.set_ylabel('Test accuracy')
        ax.set_title(name)
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle('Binary logistic regression coresets')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print('\nSaved plot to {}'.format(output_path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='5,15,30,50',
                        help='comma-separated coreset sizes to sweep')
    parser.add_argument('--seeds', type=int, default=3, help='number of random seeds to average over')
    parser.add_argument('--outer-iters', type=int, default=20,
                        help='outer (weight) iterations for BiCo w/ Weights; paper uses 150')
    parser.add_argument('--warm-inner-it', type=int, default=10,
                        help='inner GD steps performed after each weight update')
    parser.add_argument('--max-inner-it', type=int, default=150,
                        help='inner GD steps to fit the logistic regression on the current support')
    parser.add_argument('--cg-iters', type=int, default=50,
                        help='conjugate-gradient steps for the implicit gradient; paper uses 100')
    parser.add_argument('--output', default=None, help='output PNG path')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='forward-selection batch size b (default: adaptive, ~m/10; '
                             'use 1 for a faithful one-by-one reproduction of Figure 6)')
    parser.add_argument('--dataset', choices=['sklearn', 'fashion', 'cifar10', 'kmnist', 'all'], default='sklearn',
                        help='"sklearn" (default, offline, breast_cancer+digits), '
                             '"fashion" (real FashionMNIST, needs torchvision+internet), '
                             '"cifar10" (real CIFAR-10, needs torchvision+internet -- run as a '
                             'separate invocation from "fashion", not combined; see module docstring), '
                             '"kmnist" (real KMNIST, needs torchvision+internet -- also its own '
                             'separate invocation, not combined; see module docstring), '
                             '"all" (sklearn + fashion only, on one combined figure -- cifar10/kmnist '
                             'are never included in "all")')
    parser.add_argument('--fashion-classes', default='2,6',
                        help='the two FashionMNIST class indices to use (default: 2,6 = Pullover vs Shirt)')
    parser.add_argument('--fashion-max-samples', type=int, default=4000,
                        help='cap on the number of FashionMNIST points used (subsampled if larger)')
    parser.add_argument('--cifar10-classes', default='3,5',
                        help='the two CIFAR-10 class indices to use (default: 3,5 = Cat vs Dog)')
    parser.add_argument('--cifar10-max-samples', type=int, default=4000,
                        help='cap on the number of CIFAR-10 points used (subsampled if larger)')
    parser.add_argument('--kmnist-classes', default='0,1',
                        help='the two KMNIST class indices to use (default: 0,1 -- arbitrary placeholder, '
                             'see module docstring)')
    parser.add_argument('--kmnist-max-samples', type=int, default=4000,
                        help='cap on the number of KMNIST points used (subsampled if larger)')
    return parser.parse_args()


def main():
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(',')]
    fashion_classes = tuple(int(c) for c in args.fashion_classes.split(','))
    cifar10_classes = tuple(int(c) for c in args.cifar10_classes.split(','))
    kmnist_classes = tuple(int(c) for c in args.kmnist_classes.split(','))

    import os
    default_name = 'logistic_regression_coreset_accuracy_{}.png'.format(args.dataset)
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), default_name)

    datasets = load_datasets(which=args.dataset, fashion_classes=fashion_classes,
                             fashion_max_samples=args.fashion_max_samples,
                             cifar10_classes=cifar10_classes,
                             cifar10_max_samples=args.cifar10_max_samples,
                             kmnist_classes=kmnist_classes,
                             kmnist_max_samples=args.kmnist_max_samples)
    all_results = {}
    for name, (X, y) in datasets.items():
        accs, full_acc = run_dataset(name, X, y, sizes, args.seeds, args)
        all_results[name] = (accs, full_acc)

    plot_results(all_results, sizes, output_path)


if __name__ == '__main__':
    main()
