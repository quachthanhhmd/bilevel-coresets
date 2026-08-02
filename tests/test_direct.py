"""Algorithm 1 run directly on the target model (Sec. 3.3 / 3.5.1)."""

import numpy as np
import torch
import torch.nn as nn

from bicoreset import losses
from bicoreset.direct import BilevelCoreset


# ----------------------------------------------------------------------
# a ridge regression target model whose inner problem has a closed form,
# so that theta*(w) is exact and dG/dw can be checked by finite differences
# ----------------------------------------------------------------------
REG = 0.5


def _ridge_solution(X, y, w, reg=REG):
    n_s = X.shape[0]
    a = (2.0 / n_s) * (X.t() @ (w.unsqueeze(1) * X)) + reg * torch.eye(X.shape[1], dtype=X.dtype)
    b = (2.0 / n_s) * (X.t() @ (w * y))
    return torch.linalg.solve(a, b)


def _make_builder(dim, ihvp='exact'):
    def model_fn():
        m = nn.Linear(dim, 1, bias=False)
        m.double()
        return m

    def train_fn(model, X, y, weights):
        theta = _ridge_solution(X.double(), y.double(), weights.double())
        with torch.no_grad():
            model.weight.copy_(theta.reshape(1, -1))

    return BilevelCoreset(
        model_fn=model_fn,
        loss_fn=losses.mse,
        inner_reg=REG,
        ihvp=ihvp,
        train_fn=train_fn,
        max_outer_it=0,
        retrain_from_scratch=True,
        verbose=False)


def _outer_objective(X, y, theta):
    pred = (X @ theta)
    return float(torch.mean((pred - y) ** 2))


def test_implicit_gradient_matches_finite_differences():
    """Checks Eq. (3)/(5): dG/dw_k = -(dg/dtheta)^T H^-1 grad_theta l_k."""
    torch.manual_seed(0)
    n, d, s = 30, 4, 5
    X = torch.randn(n, d, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    coreset = np.arange(s)
    w = torch.ones(s, dtype=torch.float64)

    builder = _make_builder(d, ihvp='exact')
    model = builder._fit(X, y, coreset, w)
    analytic = builder._implicit_grads(model, X, y, coreset, w, X, y, coreset, scale=1.0 / s)

    eps = 1e-5
    numeric = np.zeros(s)
    for k in range(s):
        plus, minus = w.clone(), w.clone()
        plus[k] += eps
        minus[k] -= eps
        g_plus = _outer_objective(X, y, _ridge_solution(X[coreset], y[coreset], plus))
        g_minus = _outer_objective(X, y, _ridge_solution(X[coreset], y[coreset], minus))
        numeric[k] = (g_plus - g_minus) / (2 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-7)


def test_implicit_gradient_is_consistent_across_ihvp_solvers():
    torch.manual_seed(1)
    n, d, s = 25, 3, 6
    X = torch.randn(n, d, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    coreset = np.arange(s)
    w = torch.ones(s, dtype=torch.float64)

    exact = _make_builder(d, ihvp='exact')
    model = exact._fit(X, y, coreset, w)
    ref = exact._implicit_grads(model, X, y, coreset, w, X, y, np.arange(n), scale=1.0 / s)

    cg = _make_builder(d, ihvp='cg')
    cg.ihvp.max_iter, cg.ihvp.tol = 500, 1e-14
    got_cg = cg._implicit_grads(model, X, y, coreset, w, X, y, np.arange(n), scale=1.0 / s)
    np.testing.assert_allclose(got_cg, ref, rtol=1e-4, atol=1e-7)

    from bicoreset.ihvp import NeumannInverseHVP
    neumann = _make_builder(d, ihvp='exact')
    neumann.ihvp = NeumannInverseHVP(num_terms=2000, alpha=0.05)
    got_neumann = neumann._implicit_grads(model, X, y, coreset, w, X, y, np.arange(n), scale=1.0 / s)
    np.testing.assert_allclose(got_neumann, ref, rtol=1e-2, atol=1e-5)


def test_selection_rule_picks_the_argmax_of_eq5():
    """The chosen atom must be the one with the smallest implicit gradient."""
    torch.manual_seed(2)
    n, d = 40, 3
    X = torch.randn(n, d, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    builder = _make_builder(d, ihvp='exact')
    inds, _ = builder.build(X, y, 4, strategy='forward', start_size=3)

    coreset = inds[:3]
    w = torch.ones(len(coreset), dtype=torch.float64)
    model = builder._fit(X, y, coreset, w)
    candidates = np.setdiff1d(np.arange(n), coreset)
    scores = builder._implicit_grads(model, X, y, coreset, w, X, y, candidates)
    assert inds[3] == candidates[int(np.argmin(scores))]


# ----------------------------------------------------------------------
# classification: shapes, strategies, base indices, weighted coresets
# ----------------------------------------------------------------------
def _blobs(n_per_class=40, d=5, n_classes=3, sep=6.0, seed=0):
    rs = np.random.RandomState(seed)
    centers = rs.randn(n_classes, d) * sep
    X, y = [], []
    for c in range(n_classes):
        X.append(rs.randn(n_per_class, d) + centers[c])
        y.append(np.full(n_per_class, c))
    X = torch.from_numpy(np.concatenate(X)).float()
    y = torch.from_numpy(np.concatenate(y)).long()
    return X, y


def _classification_builder(d, n_classes, **kwargs):
    import models

    defaults = dict(
        model_fn=lambda: models.LogisticRegression(d, n_classes),
        loss_fn=losses.cross_entropy,
        inner_reg=1e-3,
        ihvp='cg',
        ihvp_kwargs={'max_iter': 30, 'damping': 1e-3},
        max_inner_it=60,
        inner_lr=0.1,
        max_outer_it=0,
        verbose=False)
    defaults.update(kwargs)
    return BilevelCoreset(**defaults)


def test_forward_selection_covers_all_classes():
    X, y = _blobs()
    builder = _classification_builder(5, 3)
    inds, w = builder.build(X, y, 9, strategy='forward', start_size=1)
    assert len(inds) == 9
    assert len(np.unique(inds)) == 9
    assert np.all(w == 1.0)
    # a good summary of three well separated blobs must touch every class
    assert len(np.unique(y[inds].numpy())) == 3


def test_selection_in_batches_returns_the_requested_size():
    X, y = _blobs()
    builder = _classification_builder(5, 3, candidate_pool_size=50)
    inds, _ = builder.build(X, y, 12, strategy='forward', selection_batch_size=4, start_size=2)
    assert len(inds) == 12
    assert len(np.unique(inds)) == 12


def test_elimination_and_exchange_strategies():
    X, y = _blobs(n_per_class=10)
    builder = _classification_builder(5, 3, max_inner_it=25)
    inds, _ = builder.build(X, y, 6, strategy='elimination', selection_batch_size=6)
    assert len(inds) == 6
    inds, _ = builder.build(X, y, 6, strategy='exchange', selection_batch_size=2,
                            n_exchange_steps=2)
    assert len(inds) == 6
    assert len(np.unique(inds)) == 6


def test_base_inds_are_kept_and_never_reselected():
    X, y = _blobs()
    base = np.array([0, 1, 2])
    builder = _classification_builder(5, 3)
    inds, _ = builder.build(X, y, 4, base_inds=base, start_size=1)
    assert len(inds) == 7
    assert set(base).issubset(set(inds.tolist()))
    assert len(np.unique(inds)) == 7


def test_weighted_coreset_returns_nonnegative_weights():
    X, y = _blobs(n_per_class=15)
    builder = _classification_builder(5, 3, max_outer_it=2, outer_lr=0.05,
                                      warm_inner_it=5, max_inner_it=25)
    inds, w = builder.build(X, y, 5, strategy='forward', start_size=2)
    assert len(inds) == len(w) == 5
    assert np.all(w >= 0)
    assert not np.allclose(w, 1.0)  # the weights were actually optimized


def test_minibatched_and_stochastic_hessian_paths():
    """Covers inner_batch_size / outer_batch_size / hessian_batch_size / pool subsampling."""
    X, y = _blobs(n_per_class=20)
    builder = _classification_builder(
        5, 3, max_inner_it=30, inner_batch_size=16, outer_batch_size=32,
        hessian_batch_size=4, candidate_pool_size=20, candidate_chunk_size=7)
    inds, _ = builder.build(X, y, 6, strategy='forward', selection_batch_size=2)
    assert len(inds) == 6 and len(np.unique(inds)) == 6


def test_identity_ihvp_reproduces_the_taylor_selection_rule():
    """With H^-1 ~ I the score is the plain gradient inner product (GLISTER)."""
    torch.manual_seed(3)
    n, d, s = 20, 3, 4
    X = torch.randn(n, d, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    builder = _make_builder(d, ihvp='identity')
    coreset = np.arange(s)
    w = torch.ones(s, dtype=torch.float64)
    model = builder._fit(X, y, coreset, w)
    scores = builder._implicit_grads(model, X, y, coreset, w, X, y, np.arange(n))

    # reference: -<dg/dtheta, grad_theta l_k>
    theta = model.weight.detach().reshape(-1)
    resid_out = (X @ theta - y)
    outer_grad = (2.0 / n) * (X.t() @ resid_out)
    per_sample = 2.0 * resid_out.unsqueeze(1) * X
    expected = -(per_sample @ outer_grad).numpy()
    np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-9)


def test_separate_outer_objective_is_used():
    """Sec. 5.1 instantiates g on train + validation; the API allows any outer set."""
    X, y = _blobs(n_per_class=20)
    X_out, y_out = _blobs(n_per_class=5, seed=7)
    builder = _classification_builder(5, 3, max_inner_it=30)
    inds, _ = builder.build(X, y, 5, X_outer=X_out, y_outer=y_out, start_size=1)
    assert len(inds) == 5


def test_score_candidates_api():
    X, y = _blobs(n_per_class=10)
    builder = _classification_builder(5, 3)
    cand, scores = builder.score_candidates(X, y, np.arange(5))
    assert len(cand) == len(scores) == X.shape[0] - 5
