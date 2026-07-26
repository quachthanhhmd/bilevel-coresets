"""Unsupervised coresets for Gaussian mixture models (Sec. 5.2.1, Figures 4-5).

Relative error of the negative log-likelihood obtained by fitting a GMM on
subsets of different sizes, compared to fitting on the full data set.  The
bilevel coreset first picks points that represent the modes, then refines the
covariance and weight estimates.

Run with::

    python demos/demo_gmm_coreset.py
"""

import numpy as np

from _common import set_seed

from bicoreset.gmm import GMMCoreset, WeightedGMM


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


def main():
    set_seed(0)
    k = 5
    X = synthetic_mixture(n=500)
    print('data: {} points, {} components'.format(X.shape[0], k))

    print('\n{:>8} {:>14} {:>14}'.format('m', 'uniform', 'bilevel'))
    for m in (20, 30, 50, 80):
        set_seed(0)
        builder = GMMCoreset(n_components=k, em_iters=60, cg_iters=30, damping=1e-2,
                             seed=0, verbose=False)
        inds, _ = builder.build(X, m=m, start_size=10)
        bilevel = GMMCoreset.relative_nll_error(X, inds, n_components=k, seed=0)
        uniform = np.mean([
            GMMCoreset.relative_nll_error(
                X, np.random.RandomState(s).choice(X.shape[0], m, replace=False),
                n_components=k, seed=0)
            for s in range(3)])
        print('{:>8} {:>14.5f} {:>14.5f}'.format(m, uniform, bilevel))

    full = WeightedGMM(k, seed=0).fit(X)
    print('\nfull-data NLL: {:.3f}'.format(full.nll(X)))
    print('component means:\n{}'.format(np.round(full.mu, 2)))


if __name__ == '__main__':
    main()
