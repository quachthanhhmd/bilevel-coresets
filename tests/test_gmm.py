"""Unsupervised coresets for Gaussian mixture models (Sec. 5.2.1)."""

import numpy as np
import torch

from bicoreset.gmm import GMMCoreset, TorchGMM, WeightedGMM


def _mixture(n_per_component=60, seed=0):
    rs = np.random.RandomState(seed)
    centers = np.array([[-6.0, 0.0], [6.0, 0.0], [0.0, 7.0]])
    X = np.concatenate([rs.randn(n_per_component, 2) * 0.6 + c for c in centers])
    return X


def test_weighted_em_recovers_the_component_means():
    X = _mixture()
    gmm = WeightedGMM(n_components=3, seed=0).fit(X)
    found = np.sort(gmm.mu[:, 0])
    np.testing.assert_allclose(found, np.array([-6.0, 0.0, 6.0]), atol=0.5)


def test_sample_weights_move_the_fit():
    X = _mixture(n_per_component=40)
    w = np.ones(X.shape[0])
    w[:40] = 0.0  # drop the left cluster
    gmm = WeightedGMM(n_components=2, seed=0).fit(X, w)
    assert np.min(gmm.mu[:, 0]) > -3.0


def test_torch_gmm_matches_the_em_log_likelihood():
    X = _mixture(n_per_component=30)
    gmm = WeightedGMM(n_components=3, seed=0).fit(X)
    torch_gmm = TorchGMM(3, 2).load_from_em(gmm)
    got = torch_gmm(torch.from_numpy(X).float()).detach().numpy()
    np.testing.assert_allclose(got, gmm.log_prob(X), rtol=1e-3, atol=1e-3)


def test_torch_gmm_is_twice_differentiable():
    X = torch.from_numpy(_mixture(n_per_component=10)).float()
    model = TorchGMM(2, 2)
    loss = -model(X).sum()
    params = list(model.parameters())
    from bicoreset.ihvp import hvp
    v = torch.randn(sum(p.numel() for p in params))
    out = hvp(loss, params, v)
    assert out.shape == v.shape and torch.all(torch.isfinite(out))


def test_coreset_selection_returns_the_requested_size():
    X = _mixture(n_per_component=25)
    builder = GMMCoreset(n_components=3, em_iters=25, cg_iters=15, seed=0, verbose=False)
    inds, w = builder.build(X, m=15, start_size=6)
    assert len(inds) == 15
    assert len(np.unique(inds)) == 15
    assert np.all(w == 1.0)


def test_coreset_covers_every_mode():
    X = _mixture(n_per_component=25)
    builder = GMMCoreset(n_components=3, em_iters=25, cg_iters=15, seed=0, verbose=False)
    inds, _ = builder.build(X, m=18, start_size=6)
    component = inds // 25  # the three blobs are contiguous in X
    assert len(np.unique(component)) == 3


def test_relative_nll_error_is_finite():
    X = _mixture(n_per_component=20)
    err = GMMCoreset.relative_nll_error(X, np.arange(0, 60, 2), n_components=3, seed=0)
    assert np.isfinite(err) and err >= 0
