"""Literal reproduction of Sec. 5.2.1 (Figures 4 and 5): coresets for a
Gaussian mixture model.

Paper setup, quoted from the text (no Appendix cross-reference for this
subsection -- everything needed is in the main text):

* "We generate a synthetic two-dimensional data set [...] We fit a k = 5-
  component Gaussian mixture model to the data by minimizing the loss using
  the EM algorithm."
* "To generate the coreset, we use the one-by-one forward selection with
  binary coreset weights, starting from a random sample of 10, and
  approximate the inverse Hessian-vector product via conjugate gradients."
* Figure 4: contour plots of the fitted mixture for coreset sizes shown at
  "20, 30, 50" ("A coreset of size 30 already provides accurate mean and
  covariance estimates").
* Figure 5: relative NLL error vs. subset size, x-axis spanning 20 to 100,
  comparing Uniform, "coresets for GMM generated via the sensitivity
  framework (Lucic et al., 2017)", and the bilevel coreset ("Our coreset
  construction outperforms other methods by an order of magnitude").

This is a synthetic 2-D toy experiment -- the only truly "free" one of the
five missing reproductions in terms of compute (a full run takes seconds on
CPU), which is also why the paper reports it as a qualitative illustration
rather than a table.

Deviations / choices not specified by the paper text:
* The exact means/covariances of the 5 synthetic components (the paper
  only shows the resulting contour plot, not the generating parameters).
  We reuse the well-separated 5-component layout already used by
  ``demos/demo_gmm_coreset.py`` in this repo, which qualitatively matches
  Figure 4 (five separated blobs of different shapes/orientations).
* The exact sensitivity formula of Lucic et al. (2017) for GMMs bounds each
  point's sensitivity via a bicriteria k-means solution; we approximate this
  with the same reduction already used for the classification "Sensitivity
  Coreset" baseline elsewhere in this repo (``cl_streaming/summary.py``):
  normalized squared distance to the nearest of k cluster centers plus a
  uniform per-cluster term, then importance sampling. This is the standard
  practical reduction from sensitivity bounds to a coreset (Bachem et al.,
  2018) and is asymptotically the same construction Lucic et al. use for
  GMMs (their sensitivity bound is itself derived from a bicriteria k-means
  clustering).

Run with::

    python experiments/run_gmm_paper.py
    python experiments/run_gmm_paper.py --sizes 20,30,40,50,60,70,80,90,100 --trials 5
"""

import argparse
import os

import numpy as np

from bicoreset.gmm import GMMCoreset, WeightedGMM


# ----------------------------------------------------------------------
# synthetic data set (Figure 4's 5-component 2-D mixture)
# ----------------------------------------------------------------------
def synthetic_mixture(n=600, seed=0):
    rs = np.random.RandomState(seed)
    means = np.array([[-6.0, -2.0], [6.0, 1.0], [0.0, 7.0], [-4.0, 6.0], [5.0, -6.0]])
    covs = [np.array([[1.0, 0.6], [0.6, 1.0]]),
            np.array([[2.0, -0.8], [-0.8, 0.7]]),
            np.array([[0.6, 0.0], [0.0, 2.2]]),
            np.array([[1.2, 0.4], [0.4, 1.2]]),
            np.array([[0.8, -0.3], [-0.3, 0.8]])]
    per = n // len(means)
    return np.concatenate([rs.multivariate_normal(m, c, per) for m, c in zip(means, covs)])


# ----------------------------------------------------------------------
# sensitivity-sampling baseline (Lucic et al., 2017), same reduction as
# cl_streaming/summary.py's SensitivitySampling
# ----------------------------------------------------------------------
def sensitivity_coreset_gmm(X, m, k=5, rs=None):
    """Bicriteria-k-means sensitivity sampling for GMM coresets (Lucic et al., 2017)."""
    from sklearn.cluster import KMeans
    rs = np.random.RandomState(0) if rs is None else rs
    km = KMeans(n_clusters=k, n_init=10, random_state=rs.randint(2 ** 31)).fit(X)
    assignment = km.labels_
    centers = km.cluster_centers_
    dists_sq = np.sum((X - centers[assignment]) ** 2, axis=1)
    cluster_count = np.bincount(assignment, minlength=k).astype(np.float64)
    total_cost = np.sum(dists_sq) + 1e-12
    sensitivity = dists_sq / total_cost + 1.0 / (cluster_count[assignment] + 1e-12)
    probs = sensitivity / np.sum(sensitivity)
    return rs.choice(X.shape[0], m, replace=False, p=probs)


# ----------------------------------------------------------------------
# Figure 5: relative NLL error vs subset size
# ----------------------------------------------------------------------
def run_figure5(X, k, sizes, trials, em_iters, em_restarts, cg_iters, damping, start_size, seed, out_dir):
    rows = []
    for m in sizes:
        bilevel_errs, uniform_errs, sensitivity_errs = [], [], []
        for t in range(trials):
            trial_seed = seed + t
            rs = np.random.RandomState(trial_seed)

            builder = GMMCoreset(n_components=k, em_iters=em_iters, em_restarts=em_restarts,
                                 cg_iters=cg_iters, damping=damping, seed=trial_seed, verbose=False)
            inds, _ = builder.build(X, m=m, start_size=start_size)
            bilevel_errs.append(GMMCoreset.relative_nll_error(X, inds, n_components=k, seed=trial_seed,
                                                               n_init=em_restarts))

            uni_inds = rs.choice(X.shape[0], m, replace=False)
            uniform_errs.append(GMMCoreset.relative_nll_error(X, uni_inds, n_components=k, seed=trial_seed,
                                                               n_init=em_restarts))

            sens_inds = sensitivity_coreset_gmm(X, m, k=k, rs=rs)
            sensitivity_errs.append(GMMCoreset.relative_nll_error(X, sens_inds, n_components=k, seed=trial_seed,
                                                                   n_init=em_restarts))

        row = dict(size=m,
                  uniform_mean=float(np.mean(uniform_errs)), uniform_std=float(np.std(uniform_errs)),
                  sensitivity_mean=float(np.mean(sensitivity_errs)), sensitivity_std=float(np.std(sensitivity_errs)),
                  bilevel_mean=float(np.mean(bilevel_errs)), bilevel_std=float(np.std(bilevel_errs)))
        rows.append(row)
        print('size={:>4}  uniform={:.5f}  sensitivity={:.5f}  bilevel={:.5f}'.format(
            m, row['uniform_mean'], row['sensitivity_mean'], row['bilevel_mean']))
    return rows


def plot_figure5(rows, out_path):
    import matplotlib.pyplot as plt
    sizes = [r['size'] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key, label, marker in [('uniform', 'Uniform', 'o'),
                               ('sensitivity', 'Sensitivity Coreset', 's'),
                               ('bilevel', 'Bilevel Coreset', '^')]:
        means = np.array([r[key + '_mean'] for r in rows])
        ax.plot(sizes, np.maximum(means, 1e-12), marker=marker, label=label)
    ax.set_yscale('log')
    ax.set_xlabel('Subset Size')
    ax.set_ylabel('Relative Error for NLL')
    ax.set_title('GMM coresets (k=5), relative NLL error')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print('saved {}'.format(out_path))


def plot_figure4(X, k, sizes, em_iters, em_restarts, cg_iters, damping, start_size, seed, out_path):
    import matplotlib.pyplot as plt

    full = WeightedGMM(k, seed=seed, n_init=em_restarts).fit(X)

    fig, axes = plt.subplots(1, len(sizes) + 1, figsize=(4 * (len(sizes) + 1), 4))
    xg = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 150)
    yg = np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 150)
    XX, YY = np.meshgrid(xg, yg)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

    def draw(ax, gmm, title, chosen=None):
        Z = gmm.log_prob(grid).reshape(XX.shape)
        ax.contour(XX, YY, Z, levels=12)
        ax.scatter(X[:, 0], X[:, 1], s=4, c='lightgray', alpha=0.5)
        if chosen is not None:
            ax.scatter(X[chosen, 0], X[chosen, 1], s=25, c='red', marker='x')
        ax.set_title(title)

    draw(axes[0], full, 'Full data set (n={})'.format(X.shape[0]))

    for ax, m in zip(axes[1:], sizes):
        builder = GMMCoreset(n_components=k, em_iters=em_iters, em_restarts=em_restarts,
                             cg_iters=cg_iters, damping=damping, seed=seed, verbose=False)
        inds, _ = builder.build(X, m=m, start_size=start_size)
        sub = WeightedGMM(k, seed=seed, n_init=em_restarts).fit(X[inds])
        draw(ax, sub, 'Coreset size {}'.format(m), chosen=inds)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print('saved {}'.format(out_path))


def main():
    parser = argparse.ArgumentParser(description='Sec. 5.2.1 GMM coreset reproduction (Figures 4-5)')
    parser.add_argument('--n', type=int, default=600, help='synthetic data set size')
    parser.add_argument('--k', type=int, default=5, help='number of GMM components (paper: 5)')
    parser.add_argument('--sizes', default='20,30,40,50,60,70,80,90,100',
                        help='coreset sizes for Figure 5 (paper x-axis spans 20 to 100)')
    parser.add_argument('--contour-sizes', default='20,30,50',
                        help='coreset sizes shown in Figure 4 (paper: 20, 30, 50)')
    parser.add_argument('--trials', type=int, default=5,
                        help='independent trials per size for Figure 5 (not specified by the paper; '
                             '5 gives a reasonably tight curve at negligible cost)')
    parser.add_argument('--start-size', type=int, default=10, help='paper: random sample of 10')
    parser.add_argument('--em-iters', type=int, default=100)
    parser.add_argument('--em-restarts', type=int, default=5,
                        help='random restarts for the (weighted) EM fit -- EM on a handful of '
                             'points has many poor local optima; not specified by the paper')
    parser.add_argument('--cg-iters', type=int, default=50)
    parser.add_argument('--damping', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    X = synthetic_mixture(n=args.n, seed=args.seed)
    print('synthetic 2D GMM data set: {} points, k={}'.format(X.shape[0], args.k))

    sizes = [int(s) for s in args.sizes.split(',')]
    contour_sizes = [int(s) for s in args.contour_sizes.split(',')]

    print('\n=== Figure 5: relative NLL error vs subset size ===')
    rows = run_figure5(X, args.k, sizes, args.trials, args.em_iters, args.em_restarts,
                       args.cg_iters, args.damping, args.start_size, args.seed, out_dir)
    plot_figure5(rows, os.path.join(out_dir, 'gmm_relative_nll_error.png'))

    print('\n=== Figure 4: contour plots ===')
    plot_figure4(X, args.k, contour_sizes, args.em_iters, args.em_restarts,
                args.cg_iters, args.damping, args.start_size, args.seed,
                os.path.join(out_dir, 'gmm_contours.png'))

    import csv
    csv_path = os.path.join(out_dir, 'gmm_relative_nll_error.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'uniform_mean', 'uniform_std', 'sensitivity_mean', 'sensitivity_std',
                         'bilevel_mean', 'bilevel_std'])
        for r in rows:
            writer.writerow([r['size'], r['uniform_mean'], r['uniform_std'], r['sensitivity_mean'],
                             r['sensitivity_std'], r['bilevel_mean'], r['bilevel_std']])
    print('saved {}'.format(csv_path))


if __name__ == '__main__':
    main()
