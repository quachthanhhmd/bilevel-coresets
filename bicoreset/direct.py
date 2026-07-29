"""Algorithm 1 (BiCo) run *directly on the target model*, without a proxy.

This is the main contribution of the journal version of the paper:

    "We extend the framework to constructing coresets directly for the target
     models, without a proxy, and provide several ways to speed up the
     construction while maintaining its empirical effectiveness."
     -- Borsos et al., JMLR 25 (2024), Sec. 1.

The coreset selection is the cardinality-constrained bilevel problem (Eq. (1))

    min_{w >= 0, ||w||_0 <= m}  sum_i l_i(theta*(w))
    s.t.  theta*(w) in argmin_theta sum_i w_i l_i(theta)

solved by cone-constrained generalized matching pursuit (Locatello et al.,
2017): at every step the inner problem is solved for the current support, and
the atom minimizing the linearization of the outer objective is added,

    k* = argmin_k  e_k^T grad_w G(w*_{S_{t-1}})                       (Eq. (4))
       = argmax_k  grad_theta l_k(theta*)^T H^-1 grad_theta sum_i l_i(theta*)
                                                                      (Eq. (5))

where ``H`` is the Hessian of the inner objective.  All practical variants of
Sec. 3.5.1 are implemented:

* **binary coreset weights** (``max_outer_it=0``) -- removes the inner weight
  optimization loop entirely, reducing the number of implicit gradient
  evaluations to the number of selection steps;
* **inverse-Hessian-vector product approximations** -- Neumann series or
  conjugate gradients, see :mod:`bicoreset.ihvp`;
* **selection in batches** -- ``selection_batch_size=b`` adds/removes ``b``
  points per step, dividing the cost by ``b``;
* **forward selection / exchange / elimination** -- the three strategies
  compared in Sec. 5.1 (Figure 3).

With these approximations the complexity of Sec. 3.5.1 is
``O(m b^-1 (t_g m g + t_h m g + n g))``.

Loss functions must return **per-sample** losses; see :mod:`bicoreset.losses`.
"""

import copy

import numpy as np
import torch
from torch.autograd import grad

from bicoreset.ihvp import create_ihvp, flat_grad


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def _chunks(array, size):
    for start in range(0, len(array), size):
        yield array[start:start + size]


class BilevelCoreset(object):
    """Bilevel coreset construction on the target model (Algorithm 1).

    Args:
        model_fn (callable): ``model_fn() -> torch.nn.Module``.  Called to
            create a fresh target model; required when
            ``retrain_from_scratch=True`` (recommended for deep networks with
            learning-rate schedules, cf. Sec. 5.7).
        loss_fn (callable): ``loss_fn(outputs, targets) -> (n,) tensor`` of
            per-sample losses; defines both ``l_i`` in the inner objective and,
            unless ``outer_loss_fn`` is given, in the outer objective.
        outer_loss_fn (callable): per-sample loss of the outer objective ``g``.
            Defaults to ``loss_fn``.
        inner_reg (float): coefficient of the ``lambda/2 * ||theta||^2`` ridge
            added to the inner problem.  A positive value makes the inner
            problem strongly convex for linear models, which is what the
            implicit function theorem behind Eq. (3) requires.
        ihvp (str or InverseHVP): ``'neumann'``, ``'cg'``, ``'identity'``,
            ``'exact'`` or a custom solver instance.
        ihvp_kwargs (dict): keyword arguments forwarded to the solver.
        optimizer_fn (callable): ``optimizer_fn(params) -> torch.optim.Optimizer``
            used for the inner problem.  Defaults to Adam with ``inner_lr``.
        inner_lr (float): learning rate of the default inner optimizer.
        max_inner_it (int): number of gradient steps for the inner problem,
            used for every inner solve except possibly the first (see
            ``first_inner_it``).
        first_inner_it (int): number of gradient steps for the *first* inner
            solve only (when warm-starting, i.e. ``retrain_from_scratch=False``).
            Defaults to ``max_inner_it`` if not given. Matches Appendix C's
            asymmetric warm-start schedule for Sec. 3.5 ("all variants start
            with an optimization phase on the initial point set with 5e4
            iterations; then, after each step, an additional 1e4 GD iterations
            are performed") -- pass ``first_inner_it=50_000, max_inner_it=10_000``
            to reproduce it exactly.
        inner_batch_size (int): if set, the inner problem is solved with
            minibatch SGD of this size instead of full-batch gradient descent.
        train_fn (callable): optional full override of the inner solver, with
            signature ``train_fn(model, X, y, weights) -> None``.  Use this for
            deep networks that need their own schedule/augmentation pipeline.
        max_outer_it (int): number of projected gradient steps on the coreset
            weights per selection step.  ``0`` yields an unweighted (binary)
            coreset, the variant used to scale to WideResNets.
        outer_lr (float): learning rate of the weight (outer) optimizer.
        warm_inner_it (int): inner steps performed after each weight update.
        outer_grad_scale (float): optional scaling of ``dg/dtheta`` before the
            inverse-Hessian solve; only affects conditioning, not the ranking.
        outer_batch_size (int): minibatch size used to accumulate ``dg/dtheta``
            over the (potentially large) outer data set.
        candidate_chunk_size (int): number of candidate points scored per
            double-backward call.
        candidate_pool_size (int): if set, only this many candidates are drawn
            uniformly at random per selection step (the ``candidate_batch_size``
            of the original repository).
        hessian_batch_size (int): if set, Hessian-vector products are evaluated
            on a random minibatch of the current coreset of this size --
            the stochastic Hessian approximation of Sec. 5.2.3.
        retrain_from_scratch (bool): re-initialize the model before every inner
            solve.  Recommended for deep nets, wasteful for convex proxies.
        device (str): torch device.
        logging_period (int): print progress every this many selection steps.
        verbose (bool): whether to print progress.
    """

    def __init__(self,
                 model_fn,
                 loss_fn,
                 outer_loss_fn=None,
                 inner_reg=0.0,
                 ihvp='cg',
                 ihvp_kwargs=None,
                 optimizer_fn=None,
                 inner_lr=1e-3,
                 max_inner_it=200,
                 first_inner_it=None,
                 inner_batch_size=None,
                 train_fn=None,
                 max_outer_it=0,
                 outer_lr=0.01,
                 warm_inner_it=20,
                 outer_grad_scale=1.0,
                 outer_batch_size=None,
                 candidate_chunk_size=256,
                 candidate_pool_size=None,
                 hessian_batch_size=None,
                 retrain_from_scratch=True,
                 device='cpu',
                 logging_period=1,
                 verbose=True):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.outer_loss_fn = outer_loss_fn if outer_loss_fn is not None else loss_fn
        self.inner_reg = inner_reg
        self.ihvp = create_ihvp(ihvp, **(ihvp_kwargs or {}))
        self.optimizer_fn = optimizer_fn
        self.inner_lr = inner_lr
        self.max_inner_it = max_inner_it
        self.first_inner_it = first_inner_it if first_inner_it is not None else max_inner_it
        self.inner_batch_size = inner_batch_size
        self.train_fn = train_fn
        self.max_outer_it = max_outer_it
        self.outer_lr = outer_lr
        self.warm_inner_it = warm_inner_it
        self.outer_grad_scale = outer_grad_scale
        self.outer_batch_size = outer_batch_size
        self.candidate_chunk_size = candidate_chunk_size
        self.candidate_pool_size = candidate_pool_size
        self.hessian_batch_size = hessian_batch_size
        self.retrain_from_scratch = retrain_from_scratch
        self.device = device
        self.logging_period = logging_period
        self.verbose = verbose

        self.model = None
        self.history = []

    # ------------------------------------------------------------------
    # low level helpers
    # ------------------------------------------------------------------
    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _new_model(self):
        if self.model_fn is None:
            raise ValueError('model_fn is required to instantiate the target model')
        return self.model_fn().to(self.device)

    def _index(self, data, inds):
        if torch.is_tensor(inds):
            inds = inds.cpu().numpy()
        return data[inds].to(self.device)

    def _make_optimizer(self, model):
        if self.optimizer_fn is not None:
            return self.optimizer_fn(model.parameters())
        return torch.optim.Adam(model.parameters(), lr=self.inner_lr)

    def _inner_objective(self, model, X, y, weights):
        """``sum_i w_i l_i(theta) / |S| + inner_reg/2 * ||theta||^2`` (Eq. (1))."""
        losses = self.loss_fn(model(X), y)
        obj = torch.sum(losses * weights) / max(1, losses.shape[0])
        if self.inner_reg > 0:
            obj = obj + 0.5 * self.inner_reg * sum(torch.sum(p * p) for p in model.parameters())
        return obj

    def _solve_inner(self, model, X, y, weights, n_steps=None):
        """Solve ``theta*(w) = argmin_theta sum_i w_i l_i(theta)``."""
        if self.train_fn is not None:
            self.train_fn(model, X, y, weights)
            return model
        n_steps = self.max_inner_it if n_steps is None else n_steps
        optimizer = self._make_optimizer(model)
        model.train()
        n = X.shape[0]
        for _ in range(n_steps):
            if self.inner_batch_size is not None and self.inner_batch_size < n:
                batch = torch.from_numpy(
                    np.random.choice(n, self.inner_batch_size, replace=False)).to(X.device)
                xb, yb, wb = X[batch], y[batch], weights[batch]
            else:
                xb, yb, wb = X, y, weights
            optimizer.zero_grad()
            loss = self._inner_objective(model, xb, yb, wb)
            loss.backward()
            optimizer.step()
        return model

    def _inner_loss_closure(self, model, X, y, weights):
        """Callable recomputing the inner loss (optionally on a random minibatch)."""
        n = X.shape[0]

        def closure():
            if self.hessian_batch_size is not None and self.hessian_batch_size < n:
                batch = torch.from_numpy(
                    np.random.choice(n, self.hessian_batch_size, replace=False)).to(X.device)
                return self._inner_objective(model, X[batch], y[batch], weights[batch])
            return self._inner_objective(model, X, y, weights)

        return closure

    def _outer_grad(self, model, X_outer, y_outer):
        """``dg/dtheta`` with ``g = mean_i l_i(theta*)``, accumulated in minibatches."""
        params = list(model.parameters())
        n = X_outer.shape[0]
        batch_size = self.outer_batch_size or n
        total = None
        for start in range(0, n, batch_size):
            xb = X_outer[start:start + batch_size].to(self.device)
            yb = y_outer[start:start + batch_size].to(self.device)
            loss = torch.sum(self.outer_loss_fn(model(xb), yb)) / n
            g = flat_grad(grad(loss, params), detach=True)
            total = g if total is None else total + g
        return total * self.outer_grad_scale

    def _mixed_partial_scores(self, model, X, y, inds, v, scale=1.0):
        """``- v^T grad_theta l_k(theta*)`` for every ``k`` in ``inds``.

        This is the implicit gradient ``dG/dw_k`` of Eq. (3) with ``dg/dw = 0``
        and ``d^2 f / d theta d w_k = grad_theta l_k``.  The whole vector is
        obtained with one double backward per chunk instead of one backward per
        candidate: with ``u`` the (dummy) weights of the chunk,
        ``d/du_k [v^T grad_theta sum_j u_j l_j] = v^T grad_theta l_k``.
        """
        params = list(model.parameters())
        scores = []
        for chunk in _chunks(np.asarray(inds), self.candidate_chunk_size):
            xb = self._index(X, chunk)
            yb = self._index(y, chunk)
            u = torch.ones(len(chunk), device=self.device, requires_grad=True)
            losses = self.loss_fn(model(xb), yb)
            weighted = torch.sum(losses * u) * scale
            g = flat_grad(grad(weighted, params, create_graph=True, retain_graph=True))
            score = grad(g, u, grad_outputs=v, retain_graph=False)[0].detach()
            scores.append(-score.cpu())
        return torch.cat(scores).numpy()

    def _implicit_grads(self, model, X, y, coreset_inds, weights, X_outer, y_outer,
                        target_inds, scale=1.0):
        """Implicit gradient ``dG/dw_k`` for ``k`` in ``target_inds`` (Eq. (3))."""
        params = list(model.parameters())
        model.eval()
        outer_grad = self._outer_grad(model, X_outer, y_outer)
        X_s = self._index(X, coreset_inds)
        y_s = self._index(y, coreset_inds)
        closure = self._inner_loss_closure(model, X_s, y_s, weights)
        v = self.ihvp.solve(closure, params, outer_grad)
        return self._mixed_partial_scores(model, X, y, target_inds, v, scale=scale)

    def _fit(self, X, y, inds, weights, model=None):
        """Instantiate (or reuse) the model and solve the inner problem on ``inds``."""
        is_first_fit = model is None
        if is_first_fit or self.retrain_from_scratch:
            model = self._new_model()
        X_s = self._index(X, inds)
        y_s = self._index(y, inds)
        # Warm-starting (retrain_from_scratch=False): the very first solve gets
        # first_inner_it steps, every subsequent one gets max_inner_it (Appendix
        # C's asymmetric 5e4-then-1e4 schedule). Retraining from scratch every
        # time makes the distinction moot, so always use max_inner_it there.
        n_steps = self.first_inner_it if (is_first_fit and not self.retrain_from_scratch) else self.max_inner_it
        self._solve_inner(model, X_s, y_s, weights, n_steps=n_steps)
        return model

    # ------------------------------------------------------------------
    # weighted coresets (Algorithm 1, line 6)
    # ------------------------------------------------------------------
    def _optimize_weights(self, model, X, y, inds, weights, X_outer, y_outer):
        """Projected gradient descent on ``w`` restricted to the support ``inds``.

        Implements line 6 of Algorithm 1: find a local minimum of ``G(w)`` with
        ``supp(w) = S`` by gradient descent with the implicit gradient, followed
        by a projection onto the nonnegative orthant.
        """
        if self.max_outer_it <= 0:
            return model, weights
        w = weights.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=self.outer_lr)
        scale = 1.0 / max(1, len(inds))
        for _ in range(self.max_outer_it):
            grads = self._implicit_grads(model, X, y, inds, w.detach(), X_outer, y_outer,
                                         inds, scale=scale)
            optimizer.zero_grad()
            w.grad = torch.from_numpy(grads).to(dtype=w.dtype, device=w.device).clamp_(-1.0, 1.0)
            optimizer.step()
            with torch.no_grad():
                w.clamp_(min=0.0)
            # re-solve the inner problem for the updated weights (warm started)
            X_s = self._index(X, inds)
            y_s = self._index(y, inds)
            self._solve_inner(model, X_s, y_s, w.detach(), n_steps=self.warm_inner_it)
        return model, w.detach()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build(self, X, y, m,
              X_outer=None, y_outer=None,
              base_inds=None,
              selectable_inds=None,
              strategy='forward',
              selection_batch_size=1,
              start_size=1,
              init_inds=None,
              n_exchange_steps=100,
              return_model=False):
        """Build a coreset of ``m`` points for the target model.

        Args:
            X (torch.Tensor or np.ndarray): inputs, shape ``(n, ...)``.
            y (torch.Tensor or np.ndarray): targets accepted by ``loss_fn``.
            m (int): number of points to select (*on top of* ``base_inds``).
            X_outer, y_outer: data defining the outer objective ``g``.  Defaults
                to ``(X, y)``, i.e. the loss on the full data set as in Eq. (1).
                Sec. 5.1 instead uses the sum of training and validation losses.
            base_inds (np.ndarray): indices always kept in the inner problem but
                never selected/removed (e.g. the labeled pool in Eq. (10), or a
                warm-start subset).
            selectable_inds (np.ndarray): restrict the candidate pool.
            strategy (str): ``'forward'`` (Algorithm 1), ``'exchange'`` or
                ``'elimination'`` -- the three variants of Sec. 3.5.1 compared
                in Figure 3.
            selection_batch_size (int): ``b`` points added/removed per step.
            start_size (int): size of the random subset the forward selection
                starts from (line 4 of Algorithm 1).
            init_inds (np.ndarray): explicit initial selection, overrides
                ``start_size``.
            n_exchange_steps (int): number of steps for ``strategy='exchange'``.
            return_model (bool): also return the model fitted on the coreset.

        Returns:
            tuple: ``(coreset_inds, coreset_weights)``.  With ``max_outer_it=0``
            the weights are all ones (binary/unweighted coreset).
        """
        X = _to_tensor(X)
        y = _to_tensor(y)
        n = X.shape[0]
        X_outer = X if X_outer is None else _to_tensor(X_outer)
        y_outer = y if y_outer is None else _to_tensor(y_outer)

        base_inds = np.asarray([], dtype=int) if base_inds is None else np.asarray(base_inds, dtype=int)
        pool = np.arange(n) if selectable_inds is None else np.asarray(selectable_inds, dtype=int)
        pool = np.setdiff1d(pool, base_inds)
        if m > len(pool):
            raise ValueError('requested coreset size {} exceeds the candidate pool ({})'.format(m, len(pool)))

        self.history = []
        if strategy == 'forward':
            selected = self._forward(X, y, m, X_outer, y_outer, base_inds, pool,
                                     selection_batch_size, start_size, init_inds)
        elif strategy == 'elimination':
            selected = self._elimination(X, y, m, X_outer, y_outer, base_inds, pool,
                                         selection_batch_size)
        elif strategy == 'exchange':
            selected = self._exchange(X, y, m, X_outer, y_outer, base_inds, pool,
                                      selection_batch_size, n_exchange_steps, init_inds)
        else:
            raise ValueError('unknown strategy "{}"'.format(strategy))

        inds = np.concatenate([base_inds, selected]).astype(int)
        weights = torch.ones(len(inds))
        model = self._fit(X, y, inds, weights.to(self.device))
        if self.max_outer_it > 0:
            model, weights = self._optimize_weights(model, X, y, inds, weights.to(self.device),
                                                    X_outer, y_outer)
            weights = weights.cpu()
        self.model = model
        if return_model:
            return inds, weights.numpy(), model
        return inds, weights.numpy()

    # ------------------------------------------------------------------
    def _candidates(self, pool, selected):
        available = np.setdiff1d(pool, selected)
        if self.candidate_pool_size is not None and len(available) > self.candidate_pool_size:
            available = np.random.choice(available, self.candidate_pool_size, replace=False)
        return available

    def _forward(self, X, y, m, X_outer, y_outer, base_inds, pool,
                 b, start_size, init_inds):
        """Greedy forward selection in batches (Algorithm 1 + Sec. 3.5.1)."""
        if init_inds is not None:
            selected = np.asarray(init_inds, dtype=int)
        else:
            start_size = int(np.clip(start_size, 0, m))
            selected = np.random.choice(pool, start_size, replace=False) if start_size > 0 \
                else np.asarray([], dtype=int)
        model = None
        step = 0
        while len(selected) < m:
            current = np.concatenate([base_inds, selected]).astype(int)
            weights = torch.ones(len(current), device=self.device)
            if len(current) == 0:
                # nothing to fit yet: fall back to a random first atom (line 4)
                selected = np.random.choice(pool, 1, replace=False)
                continue
            model = self._fit(X, y, current, weights, model)
            if self.max_outer_it > 0:
                model, weights = self._optimize_weights(model, X, y, current, weights,
                                                        X_outer, y_outer)
            candidates = self._candidates(pool, selected)
            scores = self._implicit_grads(model, X, y, current, weights,
                                          X_outer, y_outer, candidates)
            take = int(min(b, m - len(selected), len(candidates)))
            chosen = candidates[np.argsort(scores)[:take]]
            selected = np.concatenate([selected, chosen]).astype(int)
            step += 1
            self.history.append({'size': len(selected), 'best_score': float(np.min(scores))})
            if self.verbose and step % self.logging_period == 0:
                self._log('[forward] coreset size {}, min implicit grad {:.4e}'.format(
                    len(selected), float(np.min(scores))))
        return selected

    def _elimination(self, X, y, m, X_outer, y_outer, base_inds, pool, b):
        """Elimination in batches: start from the full data, drop the least useful."""
        selected = np.array(pool, dtype=int)
        model = None
        step = 0
        while len(selected) > m:
            current = np.concatenate([base_inds, selected]).astype(int)
            weights = torch.ones(len(current), device=self.device)
            model = self._fit(X, y, current, weights, model)
            scores = self._implicit_grads(model, X, y, current, weights,
                                          X_outer, y_outer, selected)
            drop = int(min(b, len(selected) - m))
            keep_mask = np.ones(len(selected), dtype=bool)
            keep_mask[np.argsort(scores)[-drop:]] = False
            selected = selected[keep_mask]
            step += 1
            self.history.append({'size': len(selected), 'best_score': float(np.min(scores))})
            if self.verbose and step % self.logging_period == 0:
                self._log('[elimination] coreset size {}'.format(len(selected)))
        return selected

    def _exchange(self, X, y, m, X_outer, y_outer, base_inds, pool, b, n_steps, init_inds):
        """Exchange in batches (Fedorov-style "excursion", Sec. 3.5.1)."""
        if init_inds is not None:
            selected = np.asarray(init_inds, dtype=int)
        else:
            selected = np.random.choice(pool, m, replace=False)
        model = None
        for step in range(n_steps):
            current = np.concatenate([base_inds, selected]).astype(int)
            weights = torch.ones(len(current), device=self.device)
            model = self._fit(X, y, current, weights, model)
            sel_scores = self._implicit_grads(model, X, y, current, weights,
                                              X_outer, y_outer, selected)
            drop = int(min(b, len(selected)))
            keep_mask = np.ones(len(selected), dtype=bool)
            keep_mask[np.argsort(sel_scores)[-drop:]] = False
            kept = selected[keep_mask]

            candidates = self._candidates(pool, kept)
            if len(candidates) == 0:
                break
            current = np.concatenate([base_inds, kept]).astype(int)
            weights = torch.ones(len(current), device=self.device)
            model = self._fit(X, y, current, weights, model)
            cand_scores = self._implicit_grads(model, X, y, current, weights,
                                               X_outer, y_outer, candidates)
            take = int(min(drop, len(candidates)))
            chosen = candidates[np.argsort(cand_scores)[:take]]
            selected = np.concatenate([kept, chosen]).astype(int)
            self.history.append({'size': len(selected), 'best_score': float(np.min(cand_scores))})
            if self.verbose and (step + 1) % self.logging_period == 0:
                self._log('[exchange] step {}/{}'.format(step + 1, n_steps))
        return selected

    # ------------------------------------------------------------------
    def score_candidates(self, X, y, coreset_inds, X_outer=None, y_outer=None,
                         candidate_inds=None, weights=None, model=None):
        """Expose the raw implicit gradients ``dG/dw_k`` (Eq. (3)) for inspection.

        Useful for building custom acquisition rules, e.g. the batch active
        learning objective of Eq. (10).
        """
        X = _to_tensor(X)
        y = _to_tensor(y)
        X_outer = X if X_outer is None else _to_tensor(X_outer)
        y_outer = y if y_outer is None else _to_tensor(y_outer)
        coreset_inds = np.asarray(coreset_inds, dtype=int)
        candidate_inds = np.setdiff1d(np.arange(X.shape[0]), coreset_inds) \
            if candidate_inds is None else np.asarray(candidate_inds, dtype=int)
        if weights is None:
            weights = torch.ones(len(coreset_inds), device=self.device)
        if model is None:
            model = self._fit(X, y, coreset_inds, weights)
        return candidate_inds, self._implicit_grads(model, X, y, coreset_inds, weights,
                                                    X_outer, y_outer, candidate_inds)

    def clone(self):
        """Shallow copy of the constructor configuration (fresh state)."""
        return copy.copy(self)
