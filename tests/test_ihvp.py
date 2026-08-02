"""Correctness of the inverse-Hessian-vector product solvers (Sec. 3.5.1)."""

import numpy as np
import torch
import torch.nn as nn

from bicoreset.ihvp import (
    CGInverseHVP,
    ExactInverseHVP,
    IdentityInverseHVP,
    NeumannInverseHVP,
    create_ihvp,
    hvp,
)


def _quadratic_problem(d=6, n=40, reg=0.5, seed=0):
    """Ridge regression: the Hessian is ``2/n X^T X + reg I``, known in closed form."""
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    y = torch.randn(n)
    model = nn.Linear(d, 1, bias=False)

    def loss_fn():
        pred = model(X).squeeze(-1)
        return torch.mean((pred - y) ** 2) + 0.5 * reg * torch.sum(model.weight ** 2)

    hessian = 2.0 / n * (X.t() @ X) + reg * torch.eye(d)
    return model, loss_fn, hessian


def test_hvp_matches_dense_hessian():
    model, loss_fn, hessian = _quadratic_problem()
    v = torch.randn(hessian.shape[0])
    got = hvp(loss_fn(), list(model.parameters()), v)
    assert torch.allclose(got, hessian @ v, atol=1e-5)


def test_cg_solves_the_linear_system():
    model, loss_fn, hessian = _quadratic_problem()
    v = torch.randn(hessian.shape[0])
    expected = torch.linalg.solve(hessian, v)
    got = CGInverseHVP(max_iter=200, tol=1e-12).solve(loss_fn, list(model.parameters()), v)
    assert torch.allclose(got, expected, atol=1e-4)


def test_neumann_converges_to_the_inverse():
    """``alpha * sum_i (I - alpha H)^i v -> H^-1 v`` for ``alpha < 2 / lambda_max``."""
    model, loss_fn, hessian = _quadratic_problem()
    v = torch.randn(hessian.shape[0])
    expected = torch.linalg.solve(hessian, v)
    alpha = float(1.0 / torch.linalg.eigvalsh(hessian).max())
    got = NeumannInverseHVP(num_terms=400, alpha=alpha).solve(loss_fn, list(model.parameters()), v)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-4)


def test_neumann_diverges_without_scaling_is_avoided_by_alpha():
    """A too large alpha breaks the spectral radius condition; a small one does not."""
    model, loss_fn, hessian = _quadratic_problem()
    v = torch.randn(hessian.shape[0])
    expected = torch.linalg.solve(hessian, v)
    small = NeumannInverseHVP(num_terms=200, alpha=float(0.5 / torch.linalg.eigvalsh(hessian).max()))
    err_small = torch.norm(small.solve(loss_fn, list(model.parameters()), v) - expected)
    huge = NeumannInverseHVP(num_terms=200, alpha=10.0)
    err_huge = torch.norm(huge.solve(loss_fn, list(model.parameters()), v) - expected)
    assert err_small < 1e-3
    # with alpha too large the spectral radius exceeds one and the series blows up
    assert (not torch.isfinite(err_huge)) or err_huge > err_small


def test_exact_solver_matches_torch_solve():
    model, loss_fn, hessian = _quadratic_problem()
    v = torch.randn(hessian.shape[0])
    got = ExactInverseHVP().solve(loss_fn, list(model.parameters()), v)
    assert torch.allclose(got, torch.linalg.solve(hessian, v), atol=1e-4)


def test_identity_solver_is_the_taylor_approximation():
    model, loss_fn, _ = _quadratic_problem()
    v = torch.randn(6)
    got = IdentityInverseHVP().solve(loss_fn, list(model.parameters()), v)
    assert torch.allclose(got, v)


def test_stochastic_hessian_closure_is_called_every_iteration():
    model, loss_fn, _ = _quadratic_problem()
    calls = {'n': 0}

    def counting():
        calls['n'] += 1
        return loss_fn()

    NeumannInverseHVP(num_terms=7, alpha=0.1).solve(counting, list(model.parameters()), torch.randn(6))
    assert calls['n'] == 7


def test_factory():
    assert isinstance(create_ihvp('neumann', num_terms=3), NeumannInverseHVP)
    assert isinstance(create_ihvp('cg'), CGInverseHVP)
    solver = CGInverseHVP()
    assert create_ihvp(solver) is solver
