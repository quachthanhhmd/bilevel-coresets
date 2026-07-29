"""Algorithm 2 -- weighted coresets via the L_{1/2} relaxation (Sec. 3.5.3).

"BiCo Reg" in Figures 3, 6 and 7: the weights live on the simplex and a
``sum_i sqrt(w_i)`` penalty drives most of them to zero, producing weighted
summaries that are markedly more compact than binary ones.

Run with::

    python demos/demo_bico_reg.py
"""

import numpy as np

from _common import evaluate_subset, make_blobs, set_seed, train_test_split

from bicoreset import losses
from bicoreset.regularized import RegularizedBilevelCoreset


def main():
    set_seed(0)
    n_classes, dim, target = 5, 20, 12
    X, y = make_blobs(n_per_class=120, dim=dim, n_classes=n_classes, sep=0.7, noise=1.0)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y)

    import models
    builder = RegularizedBilevelCoreset(
        model_fn=lambda: models.LogisticRegression(dim, n_classes),
        loss_fn=losses.cross_entropy,
        inner_reg=1e-3,
        beta=1e-5,
        adaptive_beta=True,
        max_outer_it=150,
        outer_lr=0.02,
        max_inner_it=300,
        warm_inner_it=25,
        inner_lr=0.05,
        ihvp='cg',
        ihvp_kwargs={'max_iter': 50, 'damping': 1e-2},
        patience=3,
        logging_period=25,
        verbose=True)

    inds, weights = builder.build(X_tr, y_tr, m=target, X_outer=X_tr, y_outer=y_tr)
    print('\nselected {} points (target {}), beta ended at {:.2e}'.format(
        len(inds), target, builder.beta))
    print('weight range [{:.4f}, {:.4f}], sum {:.4f}'.format(
        weights.min(), weights.max(), weights.sum()))

    full = evaluate_subset(X_tr, y_tr, X_te, y_te, np.arange(len(y_tr)), n_classes)
    uniform = np.mean([
        evaluate_subset(X_tr, y_tr, X_te, y_te,
                        np.random.RandomState(s).choice(len(y_tr), len(inds), replace=False),
                        n_classes)
        for s in range(5)])
    weighted = evaluate_subset(X_tr, y_tr, X_te, y_te, inds, n_classes, weights=weights)
    unweighted = evaluate_subset(X_tr, y_tr, X_te, y_te, inds, n_classes)

    print('\nfull data ({} pts):        {:.4f}'.format(len(y_tr), full))
    print('uniform  ({} pts):          {:.4f}'.format(len(inds), uniform))
    print('BiCo Reg ({} pts, weighted): {:.4f}'.format(len(inds), weighted))
    print('BiCo Reg ({} pts, binary):   {:.4f}'.format(len(inds), unweighted))


if __name__ == '__main__':
    main()
