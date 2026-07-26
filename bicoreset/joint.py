"""Joint coresets for multiple models (Sec. 4.4, Eq. (11)).

A coreset built with Eq. (1) is tied to one model and one loss.  To make the
summary transferable -- e.g. for hyperparameter tuning or architecture search,
where a model-specific summary would bias the comparison -- the paper proposes
to require the subset to be a good coreset for *several* models simultaneously:

    min_{w >= 0, ||w||_0 <= m}  sum_i [ l(f_{theta_f*}(x_i), y_i)
                                        + lambda l(g_{theta_g*}(x_i), y_i) ]
    s.t. (theta_f*, theta_g*) = argmin sum_i w_i [ l(f_{theta_f}(x_i), y_i)
                                        + lambda l(g_{theta_g}(x_i), y_i) ]   (Eq. (11))

Because the inner problem decouples across models for a fixed ``w``, the
implicit gradient is simply the ``lambda``-weighted sum of the per-model
implicit gradients.  Two solvers are provided:

* ``mode='sum'`` -- the exact linearization of Eq. (11): score every candidate
  with the weighted sum of the per-model implicit gradients.
* ``mode='alternate'`` -- the heuristic the paper actually uses for the
  experiments of Table 2: "perform the selection step alternatingly for each
  model", i.e. one batch of points is selected for model 1, then for model 2,
  and so on.  Cheaper, since only one model is scored per step.

Only binary (unweighted) coresets are supported, matching Sec. 5.2.3.
"""

import numpy as np
import torch


class JointBilevelCoreset(object):
    """Build one coreset that is simultaneously good for several target models.

    Args:
        coresets (list of BilevelCoreset): one configured
            :class:`bicoreset.direct.BilevelCoreset` per target model.  Each
            carries its own ``model_fn``, loss and inner-problem settings.
        lambdas (list of float): the weights ``[1, lambda, ...]`` of Eq. (11).
            Defaults to all ones, which the paper recommends when the loss
            magnitudes are of the same order.
        mode (str): ``'alternate'`` or ``'sum'``.
        verbose (bool): print progress.
        logging_period (int): print every this many selection steps.
    """

    def __init__(self, coresets, lambdas=None, mode='alternate', verbose=True, logging_period=1):
        if len(coresets) < 1:
            raise ValueError('at least one BilevelCoreset is required')
        self.coresets = list(coresets)
        self.lambdas = [1.0] * len(self.coresets) if lambdas is None else list(lambdas)
        if len(self.lambdas) != len(self.coresets):
            raise ValueError('lambdas and coresets must have the same length')
        self.mode = mode
        self.verbose = verbose
        self.logging_period = logging_period
        self.models = []
        self.history = []

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def build(self, X, y, m,
              X_outer=None, y_outer=None,
              base_inds=None,
              selectable_inds=None,
              selection_batch_size=1,
              start_size=1,
              init_inds=None):
        """Greedy forward selection for the joint objective of Eq. (11).

        Returns:
            tuple: ``(coreset_inds, coreset_weights)`` with binary weights.
        """
        from bicoreset.direct import _to_tensor

        X = _to_tensor(X)
        y = _to_tensor(y)
        n = X.shape[0]
        X_outer = X if X_outer is None else _to_tensor(X_outer)
        y_outer = y if y_outer is None else _to_tensor(y_outer)

        base_inds = np.asarray([], dtype=int) if base_inds is None else np.asarray(base_inds, dtype=int)
        pool = np.arange(n) if selectable_inds is None else np.asarray(selectable_inds, dtype=int)
        pool = np.setdiff1d(pool, base_inds)

        if init_inds is not None:
            selected = np.asarray(init_inds, dtype=int)
        else:
            start_size = int(np.clip(start_size, 1, m))
            selected = np.random.choice(pool, start_size, replace=False)

        models = [None] * len(self.coresets)
        self.history = []
        step = 0
        while len(selected) < m:
            current = np.concatenate([base_inds, selected]).astype(int)
            candidates = self.coresets[0]._candidates(pool, selected)

            if self.mode == 'alternate':
                active = [step % len(self.coresets)]
            elif self.mode == 'sum':
                active = list(range(len(self.coresets)))
            else:
                raise ValueError('unknown mode "{}"'.format(self.mode))

            scores = np.zeros(len(candidates))
            for idx in active:
                bc = self.coresets[idx]
                weights = torch.ones(len(current), device=bc.device)
                models[idx] = bc._fit(X, y, current, weights, models[idx])
                scores = scores + self.lambdas[idx] * bc._implicit_grads(
                    models[idx], X, y, current, weights, X_outer, y_outer, candidates)

            take = int(min(selection_batch_size, m - len(selected), len(candidates)))
            chosen = candidates[np.argsort(scores)[:take]]
            selected = np.concatenate([selected, chosen]).astype(int)
            step += 1
            self.history.append({'size': len(selected), 'models': active,
                                 'best_score': float(np.min(scores))})
            if self.verbose and step % self.logging_period == 0:
                self._log('[joint/{}] coreset size {}, scored models {}'.format(
                    self.mode, len(selected), active))

        inds = np.concatenate([base_inds, selected]).astype(int)
        self.models = []
        for idx, bc in enumerate(self.coresets):
            weights = torch.ones(len(inds), device=bc.device)
            self.models.append(bc._fit(X, y, inds, weights))
        return inds, np.ones(len(inds))
