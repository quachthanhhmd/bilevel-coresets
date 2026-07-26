"""Algorithm 2: Bilevel Coresets via Regularization ("BiCo Reg", Sec. 3.5.3).

Instead of enforcing ``||w||_0 <= m`` combinatorially, the cardinality
constraint is relaxed into a sparsity-inducing penalty.  An L1 penalty does not
work for Eq. (1), because rescaling ``w`` by a common factor leaves the inner
solution unchanged; the paper therefore restricts the weights to the simplex
``Delta_n`` (so that ``||w||_1 = 1``) and penalizes with ``L_q``, ``q = 1/2``:

    min_{w in Delta_n}  sum_i l_i(theta*) + beta * sum_i sqrt(w_i)
    s.t.  theta* = argmin_theta sum_i w_i l_i(theta) + lambda ||theta||_2^2   (Eq. (8))

optimized with the implicit gradient

    beta * d(sum_i sqrt(w_i))/dw
      - (d sum_i l_i(theta*)/d theta) H^-1 (d^2 sum_i w_i l_i(theta*) / d theta d w^T)
                                                                            (Eq. (9))

followed by the Euclidean projection onto the simplex of Duchi et al. (2008)
and a mixing step with ``eps * 1_n`` for numerical stability of the ``L_{1/2}``
derivative (Algorithm 2, lines 7-8).

This produces *weighted* coresets, which Figure 3/6/7 show to be substantially
more compact than binary ones (compression ratio > 10 for logistic regression),
at the price of an implicit gradient evaluation at every step -- hence it is
practical only for simple models, as the paper notes.
"""

import numpy as np
import torch
from torch.autograd import grad

from bicoreset.ihvp import create_ihvp, flat_grad


def project_simplex(v, z=1.0):
    """Euclidean projection onto ``{w >= 0, sum(w) = z}`` (Duchi et al., 2008)."""
    n = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cumsum = torch.cumsum(u, dim=0)
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - (cumsum - z) / ind > 0
    rho = int(torch.nonzero(cond).max().item())
    theta = (cumsum[rho] - z) / (rho + 1)
    return torch.clamp(v - theta, min=0.0)


class RegularizedBilevelCoreset(object):
    """Weighted bilevel coreset via the ``L_{1/2}`` relaxation (Algorithm 2).

    Args:
        model_fn (callable): ``model_fn() -> torch.nn.Module``.
        loss_fn (callable): per-sample loss, see :mod:`bicoreset.losses`.
        outer_loss_fn (callable): per-sample outer loss; defaults to ``loss_fn``.
        inner_reg (float): ``lambda`` of Eq. (8).  Tune it first, with uniform
            weights ``w = [1/n, ..., 1/n]``, then keep it fixed (Sec. 3.5.3).
        beta (float): initial sparsity penalty; the paper starts at ``1e-7``.
        adaptive_beta (bool): double ``beta`` whenever the number of selected
            points plateaus, until the target size is reached.
        target_size (int): desired coreset size ``m``; the loop stops as soon as
            the support is at most this large.
        max_outer_it (int): maximum number of outer iterations ``T``.
        outer_lr (float): step size for the weight updates.
        outer_optimizer (str): ``'adam'`` or ``'sgd'``.
        max_inner_it (int): inner steps for the first solve.
        warm_inner_it (int): inner steps after each weight update (warm start).
        eps (float): mixing constant of line 8 of Algorithm 2.
        truncation (float): weights below this are set to zero (line 10).
        reduction (str): ``'sum'`` (as in Eq. (8)) or ``'mean'`` for the outer
            objective and the penalty scale.
        ihvp / ihvp_kwargs: inverse-Hessian-vector product solver.
        patience (int): plateau length (in iterations) that triggers a doubling
            of ``beta``.
        retrain_from_scratch (bool): re-initialize the model at every iteration.
    """

    def __init__(self,
                 model_fn,
                 loss_fn,
                 outer_loss_fn=None,
                 inner_reg=1e-4,
                 beta=1e-7,
                 adaptive_beta=True,
                 target_size=None,
                 max_outer_it=200,
                 outer_lr=0.01,
                 outer_optimizer='adam',
                 max_inner_it=300,
                 warm_inner_it=30,
                 inner_lr=1e-2,
                 optimizer_fn=None,
                 eps=1e-8,
                 truncation=1e-4,
                 reduction='sum',
                 ihvp='cg',
                 ihvp_kwargs=None,
                 patience=5,
                 retrain_from_scratch=False,
                 grad_clip=1.0,
                 device='cpu',
                 logging_period=10,
                 verbose=True):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.outer_loss_fn = outer_loss_fn if outer_loss_fn is not None else loss_fn
        self.inner_reg = inner_reg
        self.beta = beta
        self.adaptive_beta = adaptive_beta
        self.target_size = target_size
        self.max_outer_it = max_outer_it
        self.outer_lr = outer_lr
        self.outer_optimizer = outer_optimizer
        self.max_inner_it = max_inner_it
        self.warm_inner_it = warm_inner_it
        self.inner_lr = inner_lr
        self.optimizer_fn = optimizer_fn
        self.eps = eps
        self.truncation = truncation
        self.reduction = reduction
        self.ihvp = create_ihvp(ihvp, **(ihvp_kwargs or {}))
        self.patience = patience
        self.retrain_from_scratch = retrain_from_scratch
        self.grad_clip = grad_clip
        self.device = device
        self.logging_period = logging_period
        self.verbose = verbose

        self.model = None
        self.history = []

    # ------------------------------------------------------------------
    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _reduce(self, values, n):
        return torch.sum(values) if self.reduction == 'sum' else torch.sum(values) / n

    def _inner_objective(self, model, X, y, w):
        """``sum_i w_i l_i(theta) + lambda ||theta||^2`` (Eq. (8))."""
        losses = self.loss_fn(model(X), y)
        obj = torch.sum(losses * w)
        if self.inner_reg > 0:
            obj = obj + self.inner_reg * sum(torch.sum(p * p) for p in model.parameters())
        return obj

    def _solve_inner(self, model, X, y, w, n_steps):
        if self.optimizer_fn is not None:
            optimizer = self.optimizer_fn(model.parameters())
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.inner_lr)
        model.train()
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = self._inner_objective(model, X, y, w)
            loss.backward()
            optimizer.step()
        return model

    def _implicit_grad(self, model, X, y, w, X_outer, y_outer):
        """Eq. (9): the full gradient of the regularized outer objective."""
        params = list(model.parameters())
        n_out = X_outer.shape[0]

        # dg/dtheta with g = reduce_i l_i(theta*)
        outer_loss = self._reduce(self.outer_loss_fn(model(X_outer), y_outer), n_out)
        outer_grad = flat_grad(grad(outer_loss, params), detach=True)

        # v = H^-1 dg/dtheta
        def closure():
            return self._inner_objective(model, X, y, w)

        v = self.ihvp.solve(closure, params, outer_grad)

        # -v^T grad_theta l_k for all k, via one double backward
        u = torch.ones(X.shape[0], device=X.device, requires_grad=True)
        losses = self.loss_fn(model(X), y)
        g = flat_grad(grad(torch.sum(losses * u), params, create_graph=True, retain_graph=True))
        implicit = -grad(g, u, grad_outputs=v)[0].detach()

        # beta * d/dw sum_i sqrt(w_i)
        penalty_grad = self.beta * 0.5 / torch.sqrt(torch.clamp(w, min=self.eps))
        return implicit + penalty_grad

    # ------------------------------------------------------------------
    def build(self, X, y, m=None, X_outer=None, y_outer=None, return_model=False):
        """Run Algorithm 2 and return ``(coreset_inds, coreset_weights)``.

        Args:
            X, y: data; the coreset is selected out of all ``n`` points.
            m (int): target coreset size, overrides ``target_size``.
            X_outer, y_outer: outer objective data, defaults to ``(X, y)``.

        Returns:
            tuple: indices with nonzero weight and the corresponding weights
            (normalized to sum to one, as they live on the simplex).
        """
        def _prepare(data):
            if data is None:
                return None
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            return data.to(self.device)

        X, y = _prepare(X), _prepare(y)
        X_outer = X if X_outer is None else _prepare(X_outer)
        y_outer = y if y_outer is None else _prepare(y_outer)
        target = m if m is not None else self.target_size

        n = X.shape[0]
        w = torch.full((n,), 1.0 / n, device=self.device)
        model = self.model_fn().to(self.device)
        self._solve_inner(model, X, y, w, self.max_inner_it)

        if self.outer_optimizer == 'adam':
            w_var = w.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([w_var], lr=self.outer_lr)
        else:
            w_var, optimizer = None, None

        self.history = []
        best_size, plateau = n, 0
        for it in range(self.max_outer_it):
            g = self._implicit_grad(model, X, y, w, X_outer, y_outer)
            if self.grad_clip is not None:
                g = torch.clamp(g, -self.grad_clip, self.grad_clip)

            if optimizer is not None:
                w_var.data = w.clone()
                optimizer.zero_grad()
                w_var.grad = g.clone()
                optimizer.step()
                w = w_var.data.clone()
            else:
                w = w - self.outer_lr * g

            w = project_simplex(w)                       # line 7
            w = (1.0 - self.eps) * w + self.eps          # line 8

            if self.retrain_from_scratch:
                model = self.model_fn().to(self.device)
                self._solve_inner(model, X, y, w, self.max_inner_it)
            else:
                self._solve_inner(model, X, y, w, self.warm_inner_it)

            size = int(torch.sum(w >= self.truncation).item())
            self.history.append({'it': it, 'size': size, 'beta': self.beta})
            if self.verbose and (it + 1) % self.logging_period == 0:
                self._log('[BiCo Reg] it {}, support {}, beta {:.3e}'.format(it + 1, size, self.beta))

            if target is not None and size <= target:
                break
            if self.adaptive_beta:
                # "if the number of the selected coreset points was plateauing
                #  in recent iterations, then we increase the sparsity penalty
                #  by doubling beta" (Sec. 3.5.3).  A support that grows again
                # counts as no progress as well.
                if size >= best_size - max(1, int(0.01 * best_size)):
                    plateau += 1
                else:
                    plateau = 0
                if plateau >= self.patience:
                    self.beta *= 2.0
                    plateau = 0
            best_size = min(best_size, size)

        w = w.clone()
        w[w < self.truncation] = 0.0                     # line 10
        inds = torch.nonzero(w).flatten().cpu().numpy()
        weights = w[inds]
        weights = (weights / weights.sum()).cpu().numpy()
        if target is not None and len(inds) > target:
            # keep the heaviest points if beta could not shrink the support enough
            order = np.argsort(-weights)[:target]
            inds, weights = inds[order], weights[order]
            weights = weights / weights.sum()
        self.model = model
        if return_model:
            return inds, weights, model
        return inds, weights
