"""Inverse-Hessian-vector products for implicit differentiation (Sec. 3.5.1).

The implicit gradient of the bilevel objective (Eq. (3) of the paper)

    dG/dw = dg/dw - dg/dtheta * (d^2 f / d theta d theta^T)^-1 * d^2 f / d theta d w^T

requires solving ``H x = dg/dtheta`` with ``H`` the Hessian of the inner
objective.  Materializing and inverting ``H`` costs ``O(d^3)`` which is
intractable for neural networks, so the paper proposes two approximations:

* **Conjugate gradients** (Pedregosa, 2016) -- the choice used in the original
  repository, exposed here as :class:`CGInverseHVP` (implemented in pure torch
  instead of going through ``scipy``).
* **Neumann series** (Lorraine et al., 2020),

      ``H^-1 = lim_{T->inf} sum_{i=0}^{T} (I - H)^i``,

  which the journal paper uses -- with a scaling hyperparameter ``alpha`` on the
  inner loss, so the series is applied to ``(alpha * H)^-1`` -- to scale the
  construction to WideResNets with millions of parameters (Sec. 5.2.3).
  This is :class:`NeumannInverseHVP` and was missing from the repository.

:class:`IdentityInverseHVP` approximates ``H^-1`` by the identity, which the
paper notes (Sec. 2) recovers the Taylor-expansion based selection of
GLISTER (Killamsetty et al., 2021).

Every solver takes a *callable* returning the inner loss so that the Hessian can
be re-evaluated on a fresh minibatch at each iteration -- the "stochastic
Hessian" approximation mentioned in Sec. 5.2.3.
"""

import torch
from torch.autograd import grad


def flat_grad(grads, detach=False):
    """Flatten a tuple of gradients into a single vector."""
    if detach:
        return torch.cat([g.detach().reshape(-1) for g in grads])
    return torch.cat([g.reshape(-1) for g in grads])


def hvp(loss, params, v, retain_graph=True, create_graph=False):
    """Hessian-vector product ``(d^2 loss / d theta d theta^T) v``.

    Uses the double-backward trick of Pearlmutter (1994): the product costs one
    extra backward pass and never instantiates the Hessian.

    Args:
        loss (torch.Tensor): scalar loss with a graph to ``params``.
        params (list of torch.nn.Parameter): parameters to differentiate w.r.t.
        v (torch.Tensor): flat vector of size ``sum(p.numel() for p in params)``.

    Returns:
        torch.Tensor: flat vector of the same size as ``v``.
    """
    g = flat_grad(grad(loss, params, create_graph=True, retain_graph=True))
    hv = grad(g, params, grad_outputs=v, retain_graph=retain_graph, create_graph=create_graph)
    return flat_grad(hv, detach=not create_graph)


def _as_callable(loss_or_fn):
    """Accept either a loss tensor (reused) or a callable recomputing the loss."""
    if callable(loss_or_fn):
        return loss_or_fn
    return lambda: loss_or_fn


class InverseHVP(object):
    """Base class: approximately solve ``H x = v`` for the inner Hessian ``H``."""

    def solve(self, inner_loss, params, v):
        raise NotImplementedError


class NeumannInverseHVP(InverseHVP):
    """Neumann series approximation of ``H^-1 v`` (Sec. 3.5.1).

    Computes ``alpha * sum_{i=0}^{T} (I - alpha * H)^i v``, which converges to
    ``H^-1 v`` provided ``max_j |lambda_j(I - alpha * H)| < 1``.  ``alpha`` must
    therefore be smaller than ``2 / lambda_max(H)``; the paper introduces it
    exactly for this reason.

    Args:
        num_terms (int): number of terms ``T`` in the truncated series
            (the paper uses 100).
        alpha (float): scaling of the inner loss.
        damping (float): optional ridge added to the Hessian for stability,
            i.e. the series is applied to ``H + damping * I``.
    """

    def __init__(self, num_terms=100, alpha=1.0, damping=0.0):
        self.num_terms = num_terms
        self.alpha = alpha
        self.damping = damping

    def solve(self, inner_loss, params, v):
        loss_fn = _as_callable(inner_loss)
        v = v.detach()
        partial_sum = v.clone()
        term = v.clone()
        for _ in range(self.num_terms):
            loss = loss_fn()
            h_term = hvp(loss, params, term)
            if self.damping > 0:
                h_term = h_term + self.damping * term
            term = term - self.alpha * h_term
            partial_sum = partial_sum + term
        return self.alpha * partial_sum


class CGInverseHVP(InverseHVP):
    """Conjugate gradient solution of ``(H + damping I) x = v`` (Pedregosa, 2016).

    Args:
        max_iter (int): maximum number of CG steps.
        tol (float): relative residual tolerance for early stopping.
        damping (float): ridge added to the Hessian; needed when the inner
            problem is only positive semi-definite.
    """

    def __init__(self, max_iter=100, tol=1e-8, damping=0.0):
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping

    def solve(self, inner_loss, params, v):
        loss_fn = _as_callable(inner_loss)
        v = v.detach()

        def matvec(x):
            out = hvp(loss_fn(), params, x)
            if self.damping > 0:
                out = out + self.damping * x
            return out

        x = torch.zeros_like(v)
        r = v.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)
        target = self.tol * self.tol * rs_old
        for _ in range(self.max_iter):
            if rs_old <= target or rs_old == 0:
                break
            ap = matvec(p)
            denom = torch.dot(p, ap)
            if torch.abs(denom) < 1e-20:
                break
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = torch.dot(r, r)
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x


class IdentityInverseHVP(InverseHVP):
    """``H^-1 ~ scale * I``.

    Recovers the first-order/Taylor-expansion selection rule of GLISTER
    (Killamsetty et al., 2021), which the paper describes as the special case of
    its framework where the Hessian in the implicit gradient is replaced by the
    identity.  Cheap, but ignores the curvature that makes Eq. (5) a bilinear
    similarity in the inverse-Hessian metric.
    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def solve(self, inner_loss, params, v):
        return self.scale * v.detach()


class ExactInverseHVP(InverseHVP):
    """Dense ``H^-1 v`` via an explicit Hessian; only for tests / tiny models."""

    def __init__(self, damping=0.0):
        self.damping = damping

    def solve(self, inner_loss, params, v):
        loss_fn = _as_callable(inner_loss)
        loss = loss_fn()
        d = v.numel()
        g = flat_grad(grad(loss, params, create_graph=True, retain_graph=True))
        rows = []
        for i in range(d):
            row = grad(g[i], params, retain_graph=True)
            rows.append(flat_grad(row, detach=True))
        h = torch.stack(rows)
        if self.damping > 0:
            h = h + self.damping * torch.eye(d, device=h.device, dtype=h.dtype)
        return torch.linalg.solve(h, v.detach())


_IHVP_REGISTRY = {
    'neumann': NeumannInverseHVP,
    'cg': CGInverseHVP,
    'conjugate_gradient': CGInverseHVP,
    'identity': IdentityInverseHVP,
    'exact': ExactInverseHVP,
}


def create_ihvp(name_or_solver, **kwargs):
    """Factory: ``create_ihvp('neumann', num_terms=100, alpha=0.1)``."""
    if isinstance(name_or_solver, InverseHVP):
        return name_or_solver
    if callable(name_or_solver) and not isinstance(name_or_solver, str):
        return name_or_solver
    key = str(name_or_solver).lower()
    if key not in _IHVP_REGISTRY:
        raise ValueError('Unknown inverse-HVP solver "{}", available: {}'.format(
            name_or_solver, sorted(_IHVP_REGISTRY)))
    return _IHVP_REGISTRY[key](**kwargs)
