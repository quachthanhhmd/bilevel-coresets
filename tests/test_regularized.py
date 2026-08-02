"""Algorithm 2: weighted coresets via the L_{1/2} relaxation (Sec. 3.5.3)."""

import numpy as np
import torch

from bicoreset import losses
from bicoreset.regularized import RegularizedBilevelCoreset, project_simplex


def test_projection_onto_the_simplex():
    v = torch.tensor([0.5, 0.3, -0.2, 1.4])
    p = project_simplex(v)
    assert torch.all(p >= 0)
    assert abs(float(p.sum()) - 1.0) < 1e-6
    # projection of a point already on the simplex is the identity
    q = torch.tensor([0.2, 0.3, 0.5])
    assert torch.allclose(project_simplex(q), q, atol=1e-6)


def test_projection_matches_known_solution():
    # Duchi et al. (2008), example: projecting [1, 0, 0] onto the simplex is itself
    v = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(project_simplex(v), v, atol=1e-6)
    # a uniform shift leaves the projection unchanged
    a = project_simplex(torch.tensor([0.4, 0.1, 0.9]))
    b = project_simplex(torch.tensor([0.4, 0.1, 0.9]) + 3.0)
    assert torch.allclose(a, b, atol=1e-6)


def _redundant_blobs(n_per_class=30, d=4, n_classes=2, seed=0):
    rs = np.random.RandomState(seed)
    centers = rs.randn(n_classes, d) * 5.0
    X, y = [], []
    for c in range(n_classes):
        X.append(rs.randn(n_per_class, d) * 0.2 + centers[c])
        y.append(np.full(n_per_class, c))
    return (torch.from_numpy(np.concatenate(X)).float(),
            torch.from_numpy(np.concatenate(y)).long())


def _builder(d, n_classes, **kwargs):
    import models

    defaults = dict(
        model_fn=lambda: models.LogisticRegression(d, n_classes),
        loss_fn=losses.cross_entropy,
        inner_reg=1e-3,
        beta=1e-4,
        max_outer_it=40,
        outer_lr=0.05,
        max_inner_it=120,
        warm_inner_it=15,
        inner_lr=0.05,
        ihvp='cg',
        ihvp_kwargs={'max_iter': 25, 'damping': 1e-2},
        patience=2,
        verbose=False)
    defaults.update(kwargs)
    return RegularizedBilevelCoreset(**defaults)


def test_returns_a_sparse_weighted_coreset_on_the_simplex():
    X, y = _redundant_blobs()
    builder = _builder(4, 2)
    inds, w = builder.build(X, y, m=10)
    assert len(inds) == len(w)
    assert 0 < len(inds) <= 10
    assert np.all(w > 0)
    assert abs(w.sum() - 1.0) < 1e-5
    assert len(np.unique(y[inds].numpy())) == 2


def test_beta_is_doubled_when_the_support_plateaus():
    X, y = _redundant_blobs(n_per_class=15)
    builder = _builder(4, 2, beta=1e-9, adaptive_beta=True, patience=1, max_outer_it=12)
    builder.build(X, y, m=2)
    betas = [h['beta'] for h in builder.history]
    assert max(betas) > min(betas)


def test_support_shrinks_with_a_large_sparsity_penalty():
    X, y = _redundant_blobs(n_per_class=30)
    n = X.shape[0]
    builder = _builder(4, 2, beta=1e-2, adaptive_beta=False, max_outer_it=40)
    inds, _ = builder.build(X, y, m=None)
    sizes = [h['size'] for h in builder.history]
    assert sizes[-1] < max(sizes)
    assert len(inds) < n / 2
