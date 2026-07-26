"""Pool-based semi-supervised batch active learning loop (Sec. 4.3 / 5.5).

Each round: retrain the semi-supervised learner from scratch on the current
labeled pool plus the unlabeled pool, query the labels of a batch of ``m``
points chosen by the acquisition strategy, and evaluate.  Retraining from
scratch every round is what the paper found to work best -- "we found that
retraining from scratch outperformed the warm-started training [...] for all
acquisition strategies" -- and it also makes the acquisition cost negligible
compared to training.
"""

import copy

import numpy as np
import torch

from batch_active_learning.acquisition import get_acquisition_fn


class ActiveLearningLoop(object):
    """Run several rounds of batch active learning.

    Args:
        model_fn (callable): ``model_fn() -> torch.nn.Module``, re-instantiated
            every round.
        trainer (MixMatchTrainer): the semi-supervised learner.
        acquisition (str or callable): acquisition strategy, see
            :mod:`batch_active_learning.acquisition`.
        acquisition_kwargs (dict): extra arguments for the strategy (e.g.
            ``feature_fn`` for ``'bico'``).
        batch_size (int): number of labels acquired per round (``m = 10`` in
            the paper).
        rounds (int): number of acquisition rounds.
        seed (int): random seed.
        verbose (bool): print per-round accuracy.
    """

    def __init__(self, model_fn, trainer, acquisition='bico', acquisition_kwargs=None,
                 batch_size=10, rounds=5, seed=None, verbose=True):
        self.model_fn = model_fn
        self.trainer = trainer
        self.acquisition = acquisition
        self.acquisition_kwargs = acquisition_kwargs or {}
        self.batch_size = batch_size
        self.rounds = rounds
        self.rs = np.random.RandomState(seed)
        self.verbose = verbose
        self.history = []
        self.model = None

    def _acquire(self, model, X_l, y_l, X_u):
        fn = self.acquisition if callable(self.acquisition) else get_acquisition_fn(self.acquisition)
        kwargs = dict(self.acquisition_kwargs)
        kwargs.setdefault('rs', self.rs)
        return np.asarray(fn(model, self.trainer, X_l, y_l, X_u, self.batch_size, **kwargs), dtype=int)

    def run(self, X_pool, y_pool, labeled_inds, X_test=None, y_test=None):
        """Run the acquisition rounds.

        Args:
            X_pool, y_pool: the whole pool; ``y_pool`` is only revealed for the
                points that have been queried.
            labeled_inds (np.ndarray): initial labeled pool (the paper
                guarantees at least one sample per class).
            X_test, y_test: optional evaluation set.

        Returns:
            list of dict: per-round records with the labeled set size, the test
            accuracy and the acquired indices.
        """
        labeled = np.asarray(labeled_inds, dtype=int)
        n = X_pool.shape[0]
        self.history = []
        for rnd in range(self.rounds + 1):
            unlabeled = np.setdiff1d(np.arange(n), labeled)
            model = self.model_fn()
            model = self.trainer.train(model, X_pool[labeled], y_pool[labeled], X_pool[unlabeled])
            acc = None
            if X_test is not None:
                acc = self.trainer.accuracy(model, X_test, y_test)
            record = {'round': rnd, 'n_labeled': len(labeled), 'test_accuracy': acc}
            if self.verbose:
                print('[AL/{}] round {}: {} labels, test acc {}'.format(
                    self.acquisition if isinstance(self.acquisition, str) else 'custom',
                    rnd, len(labeled), 'n/a' if acc is None else '{:.4f}'.format(acc)))

            if rnd < self.rounds:
                local = self._acquire(model, X_pool[labeled], y_pool[labeled], X_pool[unlabeled])
                acquired = unlabeled[local]
                record['acquired'] = acquired
                labeled = np.concatenate([labeled, acquired]).astype(int)
            self.history.append(record)
            self.model = model
        self.labeled_inds = labeled
        return self.history

    def clone_with(self, acquisition, **kwargs):
        """Copy of the loop with a different acquisition strategy (for comparisons)."""
        other = copy.copy(self)
        other.acquisition = acquisition
        other.acquisition_kwargs = dict(self.acquisition_kwargs)
        other.acquisition_kwargs.update(kwargs)
        other.history = []
        return other
