"""Algorithm 1 directly on the target model -- the variants of Sec. 3.5.1.

Reproduces, on a small synthetic problem, the comparison of Figure 3:
uniform sampling vs. one-by-one forward selection vs. forward selection in
batches vs. exchange vs. elimination, and Neumann-series vs. conjugate-gradient
inverse-Hessian-vector products.

Run with::

    python demos/demo_direct_coreset.py
"""

import numpy as np
import torch

from _common import accuracy, evaluate_subset, make_blobs, set_seed, train_test_split

from bicoreset import losses
from bicoreset.direct import BilevelCoreset


def make_builder(dim, n_classes, ihvp='cg', ihvp_kwargs=None, **kwargs):
    import models

    defaults = dict(
        model_fn=lambda: models.LogisticRegression(dim, n_classes),
        loss_fn=losses.cross_entropy,
        inner_reg=1e-3,
        ihvp=ihvp,
        ihvp_kwargs=ihvp_kwargs or {'max_iter': 50, 'damping': 1e-3},
        max_inner_it=150,
        inner_lr=0.1,
        max_outer_it=0,
        candidate_pool_size=200,
        retrain_from_scratch=True,
        verbose=False)
    defaults.update(kwargs)
    return BilevelCoreset(**defaults)


def main():
    set_seed(0)
    n_classes, dim, m = 5, 20, 10
    X, y = make_blobs(n_per_class=200, dim=dim, n_classes=n_classes, sep=0.7, noise=1.0)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y)

    full = evaluate_subset(X_tr, y_tr, X_te, y_te, np.arange(len(y_tr)), n_classes)
    print('full data set ({} points): {:.4f}'.format(len(y_tr), full))

    uniform = np.mean([
        evaluate_subset(X_tr, y_tr, X_te, y_te,
                        np.random.RandomState(s).choice(len(y_tr), m, replace=False), n_classes)
        for s in range(5)])
    print('uniform          (m={}): {:.4f}'.format(m, uniform))

    configs = [
        ('BiCo fwd        ', dict(strategy='forward', selection_batch_size=1, start_size=1), {}),
        ('BiCo fwd b=5    ', dict(strategy='forward', selection_batch_size=5, start_size=1), {}),
        ('BiCo exchange   ', dict(strategy='exchange', selection_batch_size=3, n_exchange_steps=5), {}),
        ('BiCo elimination', dict(strategy='elimination', selection_batch_size=40), {}),
    ]
    for name, build_kwargs, builder_kwargs in configs:
        set_seed(0)
        builder = make_builder(dim, n_classes, **builder_kwargs)
        inds, w = builder.build(X_tr, y_tr, m, **build_kwargs)
        acc = evaluate_subset(X_tr, y_tr, X_te, y_te, inds, n_classes, weights=w)
        print('{} (m={}): {:.4f}'.format(name, m, acc))

    # Neumann series IHVP (Sec. 3.5.1) -- the variant that scales to WideResNets
    set_seed(0)
    builder = make_builder(dim, n_classes, ihvp='neumann',
                           ihvp_kwargs={'num_terms': 100, 'alpha': 0.05})
    inds, w = builder.build(X_tr, y_tr, m, strategy='forward', selection_batch_size=1)
    print('BiCo fwd/Neumann (m={}): {:.4f}'.format(m, evaluate_subset(
        X_tr, y_tr, X_te, y_te, inds, n_classes, weights=w)))

    # weighted coreset (Algorithm 1 with the weight optimization of line 6)
    set_seed(0)
    builder = make_builder(dim, n_classes, max_outer_it=3, outer_lr=0.05, warm_inner_it=20)
    inds, w = builder.build(X_tr, y_tr, m, strategy='forward', selection_batch_size=1)
    print('BiCo weighted    (m={}): {:.4f}  weights in [{:.2f}, {:.2f}]'.format(
        m, evaluate_subset(X_tr, y_tr, X_te, y_te, inds, n_classes, weights=w), w.min(), w.max()))


if __name__ == '__main__':
    main()
