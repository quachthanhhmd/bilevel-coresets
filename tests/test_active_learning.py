"""Semi-supervised batch active learning (Sec. 4.3 / 5.5)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from batch_active_learning.acquisition import ACQUISITION_FNS, bico, get_acquisition_fn
from batch_active_learning.active_learning import ActiveLearningLoop
from batch_active_learning.mixmatch import MixMatchTrainer, sharpen
from batch_active_learning.proxy import make_rbf_kernel, nystrom_feature_map


def _blobs(n_per_class=60, d=6, n_classes=3, seed=0):
    rs = np.random.RandomState(seed)
    centers = rs.randn(n_classes, d) * 4.0
    X = np.concatenate([rs.randn(n_per_class, d) + centers[c] for c in range(n_classes)])
    y = np.concatenate([np.full(n_per_class, c) for c in range(n_classes)])
    perm = rs.permutation(len(y))
    return torch.from_numpy(X[perm]).float(), torch.from_numpy(y[perm]).long()


def _model_fn(d=6, n_classes=3):
    import models
    return lambda: models.FNNet(d, 16, n_classes)


def _trainer(epochs=10):
    return MixMatchTrainer(num_classes=3, augment_fn=lambda x: x + 0.05 * torch.randn_like(x),
                           n_augmentations=2, epochs=epochs, batch_size=32, lr=0.02,
                           lambda_u=1.0, device='cpu')


def test_sharpen_produces_a_distribution():
    p = torch.tensor([[0.6, 0.3, 0.1]])
    q = sharpen(p, 0.5)
    assert abs(float(q.sum()) - 1.0) < 1e-6
    assert q[0, 0] > p[0, 0]  # sharpening concentrates mass on the mode


def test_mixmatch_learns_something():
    X, y = _blobs()
    trainer = _trainer(epochs=25)
    model = trainer.train(_model_fn()(), X[:30], y[:30], X[30:])
    assert trainer.accuracy(model, X, y) > 0.6


@pytest.mark.parametrize('name', sorted(ACQUISITION_FNS))
def test_every_acquisition_strategy_returns_a_valid_batch(name):
    X, y = _blobs(n_per_class=25)
    trainer = _trainer(epochs=5)
    model = trainer.train(_model_fn()(), X[:12], y[:12], X[12:])
    fn = get_acquisition_fn(name)
    kwargs = {'rs': np.random.RandomState(0)}
    if name == 'bico':
        kwargs.update(max_inner_it=60, cg_iters=20, num_classes=3)
    chosen = np.asarray(fn(model, trainer, X[:12], y[:12], X[12:], 5, **kwargs))
    assert len(chosen) == 5
    assert len(np.unique(chosen)) == 5
    assert chosen.min() >= 0 and chosen.max() < X.shape[0] - 12


def test_bico_acquisition_with_a_nystrom_proxy():
    X, y = _blobs(n_per_class=25)
    trainer = _trainer(epochs=5)
    model = trainer.train(_model_fn()(), X[:12], y[:12], X[12:])
    feature_map = nystrom_feature_map(make_rbf_kernel(gamma=0.05), X.numpy(), q=20,
                                      rs=np.random.RandomState(0))
    chosen = bico(model, trainer, X[:12], y[:12], X[12:], 4,
                  feature_fn=feature_map, num_classes=3, max_inner_it=80, cg_iters=20)
    assert len(chosen) == 4 and len(np.unique(chosen)) == 4


def test_active_learning_loop_grows_the_labeled_pool():
    X, y = _blobs(n_per_class=30)
    loop = ActiveLearningLoop(_model_fn(), _trainer(epochs=5), acquisition='uniform',
                              batch_size=5, rounds=2, seed=0, verbose=False)
    history = loop.run(X, y, labeled_inds=np.arange(9), X_test=X, y_test=y)
    assert [h['n_labeled'] for h in history] == [9, 14, 19]
    assert all(0.0 <= h['test_accuracy'] <= 1.0 for h in history)
    assert len(np.unique(loop.labeled_inds)) == 19


def test_active_learning_loop_with_bico():
    X, y = _blobs(n_per_class=25)
    loop = ActiveLearningLoop(
        _model_fn(), _trainer(epochs=5), acquisition='bico',
        acquisition_kwargs={'num_classes': 3, 'max_inner_it': 60, 'cg_iters': 20},
        batch_size=4, rounds=1, seed=0, verbose=False)
    history = loop.run(X, y, labeled_inds=np.arange(9))
    assert [h['n_labeled'] for h in history] == [9, 13]
