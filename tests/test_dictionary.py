"""Dictionary selection for compressed sensing (Sec. 4.5, Eq. (12))."""

import numpy as np
import torch
import torch.nn as nn

from bicoreset.dictionary import DictionarySelector


def _sparse_signals(n=40, d=16, sparsity=3, seed=0):
    rs = np.random.RandomState(seed)
    X = np.zeros((n, d))
    for i in range(n):
        support = rs.choice(d, sparsity, replace=False)
        X[i, support] = rs.randn(sparsity)
    return torch.from_numpy(X).float()


def _dictionary(m=24, d=16, seed=1):
    rs = np.random.RandomState(seed)
    return torch.from_numpy(rs.randn(m, d) / np.sqrt(d)).float()


def test_l2_reconstruction_matches_the_closed_form():
    X, A = _sparse_signals(), _dictionary()
    sel = DictionarySelector(A, recovery='l2', lam=0.01, verbose=False)
    w = torch.zeros(A.shape[0])
    w[:8] = 1.0
    Xhat, _ = sel.reconstruct(X, w)
    m = A.t() @ (w.unsqueeze(1) * A)
    expected = torch.linalg.solve(m + 0.01 * torch.eye(A.shape[1]), m @ X.t()).t()
    assert torch.allclose(Xhat, expected, atol=1e-5)


def test_implicit_gradient_matches_autograd_through_the_closed_form():
    """For the L2 recovery the whole bilevel objective is differentiable in w."""
    X, A = _sparse_signals(n=12, d=8), _dictionary(m=10, d=8)
    lam = 0.05
    sel = DictionarySelector(A, recovery='l2', lam=lam, damping=0.0, cg_iters=300, verbose=False)
    w = torch.rand(A.shape[0]) + 0.5

    analytic = sel.implicit_grads(X, w)

    w_var = w.clone().requires_grad_(True)
    m = A.t() @ (w_var.unsqueeze(1) * A)
    xhat = torch.linalg.solve(m + lam * torch.eye(A.shape[1]), m @ X.t()).t()
    g = torch.sum((X - xhat) ** 2) / X.shape[0]
    expected = torch.autograd.grad(g, w_var)[0]

    assert torch.allclose(analytic, expected, rtol=1e-3, atol=1e-4)


def test_error_decreases_with_more_measurements():
    X, A = _sparse_signals(), _dictionary()
    sel = DictionarySelector(A, recovery='l2', lam=0.01, verbose=False)
    errors = [sel.reconstruction_error(X, sel.select(X, k)) for k in (2, 6, 12)]
    assert errors[0] > errors[1] > errors[2]


def test_bilevel_beats_random_and_approx_greedy():
    """Averaged over problem instances, as in Figure 12."""
    k = 8
    bilevel_errs, random_errs, greedy_errs = [], [], []
    for seed in range(3):
        X = _sparse_signals(n=60, d=16, sparsity=4, seed=seed)
        A = _dictionary(m=40, d=16, seed=seed + 10)
        sel = DictionarySelector(A, recovery='l2', lam=0.01, verbose=False)
        bilevel_errs.append(sel.reconstruction_error(X, sel.select(X, k)))
        rs = np.random.RandomState(seed)
        random_errs.append(np.mean([sel.reconstruction_error(X, sel.select_random(k, rs))
                                    for _ in range(5)]))
        greedy_errs.append(sel.reconstruction_error(X, sel.select_approx_greedy(X, k)))
    assert np.mean(bilevel_errs) < np.mean(random_errs)
    assert np.mean(bilevel_errs) < np.mean(greedy_errs)


def test_l1_recovery_runs_and_selects():
    X, A = _sparse_signals(n=20, d=12), _dictionary(m=16, d=12)
    sel = DictionarySelector(A, recovery='l1', lam=0.01, ista_iters=50,
                             damping=1e-2, cg_iters=30, verbose=False)
    inds = sel.select(X, 5)
    assert len(inds) == 5 and len(np.unique(inds)) == 5
    assert np.isfinite(sel.reconstruction_error(X, inds))


def test_generative_model_recovery_runs():
    torch.manual_seed(0)
    d, p = 12, 3
    generator = nn.Sequential(nn.Linear(p, 16), nn.ReLU(), nn.Linear(16, d))
    X, A = _sparse_signals(n=8, d=d), _dictionary(m=10, d=d)
    sel = DictionarySelector(A, recovery='gm', lam=0.01, generator=generator, latent_dim=p,
                             gm_iters=30, damping=1e-2, cg_iters=20, verbose=False)
    inds = sel.select(X, 3)
    assert len(inds) == 3
    assert np.isfinite(sel.reconstruction_error(X, inds))
