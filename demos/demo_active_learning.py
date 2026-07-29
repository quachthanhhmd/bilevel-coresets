"""Semi-supervised batch active learning (Sec. 4.3 / 5.5, Figure 10).

Compares the bilevel acquisition of Eq. (10) with uniform sampling,
max-entropy, k-center, consistency and BADGE, all under the same MixMatch
semi-supervised training.  The data set is synthetic so the demo runs on a CPU
in under a minute; for the audio experiments of the paper swap in the Spoken
Digit / Speech Commands datasets and a WideResNet.

Run with::

    python demos/demo_active_learning.py
"""

import os

import numpy as np
import torch

from _common import make_blobs, set_seed, train_test_split

from batch_active_learning.active_learning import ActiveLearningLoop
from batch_active_learning.mixmatch import MixMatchTrainer
from batch_active_learning.proxy import make_rbf_kernel, nystrom_feature_map


def main():
    set_seed(0)
    dim, n_classes = 20, 8
    X, y = make_blobs(n_per_class=120, dim=dim, n_classes=n_classes, sep=0.6, noise=1.0)
    X_pool, y_pool, X_test, y_test = train_test_split(X, y, test_frac=0.3)

    def model_fn():
        import models
        return models.FNNet(dim, 32, n_classes)

    trainer = MixMatchTrainer(
        num_classes=n_classes,
        augment_fn=lambda x: x + 0.1 * torch.randn_like(x),
        n_augmentations=2, temperature=0.5, alpha=0.75, lambda_u=1.0,
        epochs=30, batch_size=32, lr=0.01, device='cpu')

    # one sample per class as the initial labeled pool (as in the paper)
    rs = np.random.RandomState(0)
    initial = np.concatenate([rs.choice(np.where(y_pool.numpy() == c)[0], 1, replace=False)
                              for c in range(n_classes)])

    feature_map = nystrom_feature_map(make_rbf_kernel(gamma=0.02), X_pool.numpy(), q=100,
                                      rs=np.random.RandomState(0))

    strategies = {
        'uniform': {},
        'max_entropy': {},
        'kcenter': {},
        'consistency': {},
        'badge': {},
        'bico': {'feature_fn': feature_map, 'num_classes': n_classes,
                 'inner_reg': 1e-4, 'max_inner_it': 300, 'cg_iters': 50},
    }

    n_seeds = int(os.environ.get('N_SEEDS', 3))
    print('averaging over {} seeds (set N_SEEDS to change)\n'.format(n_seeds))
    results, sizes = {}, None
    for name, kwargs in strategies.items():
        runs = []
        for seed in range(n_seeds):
            set_seed(seed)
            loop = ActiveLearningLoop(model_fn, trainer, acquisition=name,
                                      acquisition_kwargs=kwargs, batch_size=8, rounds=3,
                                      seed=seed, verbose=False)
            history = loop.run(X_pool, y_pool, initial, X_test, y_test)
            runs.append([h['test_accuracy'] for h in history])
            sizes = [h['n_labeled'] for h in history]
        results[name] = np.mean(runs, axis=0)
        print('{:<12} {}'.format(name, ' '.join('{:.4f}'.format(a) for a in results[name])))

    print('\nlabeled pool sizes: {}'.format(sizes))
    best = max(results, key=lambda k: results[k][-1])
    print('best after the last round: {}'.format(best))


if __name__ == '__main__':
    main()
