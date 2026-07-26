"""Joint coresets for several models (Sec. 4.4, Eq. (11)).

Mirrors Table 2: a coreset built for model A alone transfers worse to model B
than a coreset built jointly for A and B, at the price of a small degradation
on A.

Run with::

    python demos/demo_joint_coreset.py
"""

import numpy as np
import torch
import torch.nn as nn

from _common import make_blobs, set_seed, train_test_split

from bicoreset import losses
from bicoreset.direct import BilevelCoreset
from bicoreset.joint import JointBilevelCoreset


def make_linear(dim, n_classes):
    import models
    return lambda: models.LogisticRegression(dim, n_classes)


def make_mlp(dim, n_classes, hidden=16):
    return lambda: nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))


def builder(model_fn, **kwargs):
    defaults = dict(
        model_fn=model_fn,
        loss_fn=losses.cross_entropy,
        inner_reg=1e-3,
        ihvp='cg',
        ihvp_kwargs={'max_iter': 40, 'damping': 1e-3},
        max_inner_it=150,
        inner_lr=0.05,
        candidate_pool_size=200,
        verbose=False)
    defaults.update(kwargs)
    return BilevelCoreset(**defaults)


def train_and_score(model_fn, X, y, X_te, y_te, epochs=300, lr=0.05):
    model = model_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(X), y.long()).backward()
        optimizer.step()
    with torch.no_grad():
        return float((torch.argmax(model(X_te), dim=1) == y_te.long()).float().mean())


def main():
    set_seed(0)
    dim, n_classes, m = 20, 5, 15
    X, y = make_blobs(n_per_class=200, dim=dim, n_classes=n_classes, sep=0.7, noise=1.0)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y)

    linear_fn = make_linear(dim, n_classes)
    mlp_fn = make_mlp(dim, n_classes)

    set_seed(0)
    single, _ = builder(linear_fn).build(X_tr, y_tr, m, strategy='forward', start_size=1)

    set_seed(0)
    joint = JointBilevelCoreset([builder(linear_fn), builder(mlp_fn)],
                                lambdas=[1.0, 1.0], mode='alternate', verbose=False)
    joint_inds, _ = joint.build(X_tr, y_tr, m, start_size=1)

    rs = np.random.RandomState(0)
    uniform_inds = rs.choice(len(y_tr), m, replace=False)

    print('{:<20} {:>10} {:>10}'.format('summary (m={})'.format(m), 'linear', 'MLP'))
    for name, inds in [('uniform', uniform_inds),
                       ('BiCo linear only', single),
                       ('BiCo linear + MLP', joint_inds)]:
        set_seed(1)
        acc_lin = train_and_score(linear_fn, X_tr[inds], y_tr[inds], X_te, y_te)
        set_seed(1)
        acc_mlp = train_and_score(mlp_fn, X_tr[inds], y_tr[inds], X_te, y_te)
        print('{:<20} {:>10.4f} {:>10.4f}'.format(name, acc_lin, acc_mlp))


if __name__ == '__main__':
    main()
