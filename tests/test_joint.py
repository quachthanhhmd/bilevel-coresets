"""Joint coresets for several target models (Sec. 4.4, Eq. (11))."""

import numpy as np
import torch
import torch.nn as nn

from bicoreset import losses
from bicoreset.direct import BilevelCoreset
from bicoreset.joint import JointBilevelCoreset


def _blobs(n_per_class=25, d=4, n_classes=3, seed=0):
    rs = np.random.RandomState(seed)
    centers = rs.randn(n_classes, d) * 5.0
    X = np.concatenate([rs.randn(n_per_class, d) + centers[c] for c in range(n_classes)])
    y = np.concatenate([np.full(n_per_class, c) for c in range(n_classes)])
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()


def _builders(d, n_classes):
    import models

    def make(model_fn):
        return BilevelCoreset(
            model_fn=model_fn,
            loss_fn=losses.cross_entropy,
            inner_reg=1e-3,
            ihvp='cg',
            ihvp_kwargs={'max_iter': 20, 'damping': 1e-3},
            max_inner_it=40,
            inner_lr=0.1,
            verbose=False)

    linear = make(lambda: models.LogisticRegression(d, n_classes))
    mlp = make(lambda: nn.Sequential(nn.Linear(d, 8), nn.ReLU(), nn.Linear(8, n_classes)))
    return [linear, mlp]


def test_alternating_joint_selection():
    X, y = _blobs()
    joint = JointBilevelCoreset(_builders(4, 3), mode='alternate', verbose=False)
    inds, w = joint.build(X, y, 8, start_size=2)
    assert len(inds) == 8 and len(np.unique(inds)) == 8
    assert np.all(w == 1.0)
    # the selection step alternates between the two models
    used = [tuple(h['models']) for h in joint.history]
    assert (0,) in used and (1,) in used


def test_summed_joint_selection_scores_all_models():
    X, y = _blobs()
    joint = JointBilevelCoreset(_builders(4, 3), lambdas=[1.0, 0.5], mode='sum', verbose=False)
    inds, _ = joint.build(X, y, 6, start_size=2)
    assert len(inds) == 6
    assert all(h['models'] == [0, 1] for h in joint.history)
    assert len(joint.models) == 2


def test_joint_coreset_covers_all_classes():
    X, y = _blobs()
    joint = JointBilevelCoreset(_builders(4, 3), mode='sum', verbose=False)
    inds, _ = joint.build(X, y, 9, start_size=1)
    assert len(np.unique(y[inds].numpy())) == 3
