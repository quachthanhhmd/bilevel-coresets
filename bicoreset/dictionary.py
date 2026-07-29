"""Dictionary selection for compressed sensing (Sec. 4.5, Eq. (12)).

Here the coreset is built over *measurements* rather than data points: given a
dictionary ``A = [a_1, ..., a_m]`` of linear measurement vectors and a
representative signal set ``{x_i}``, we look for the ``k`` measurements that
minimize the reconstruction error,

    min_{||w||_0 <= k}  1/n sum_i ||x_i - xhat_i(w)||_2^2
    s.t.  xhat_i(w) = argmin_y sum_j w_j (a_j^T (x_i - y))^2 + lambda R(y)  (Eq. (12))

with ``w`` binary.  Three recovery operators ``R`` are supported, matching
Figure 12:

* ``'l2'``  -- ridge recovery, ``R(y) = ||y||_2^2``, closed form;
* ``'l1'``  -- basis pursuit / compressed sensing, ``R(y) = ||y||_1``, solved
  with ISTA.  The objective is not differentiable everywhere; as the paper
  notes, "using an element of the sub-differential proves to be a viable
  strategy" -- the ``L1`` term contributes zero curvature, so the Hessian used
  in the implicit gradient is that of the quadratic part (plus damping);
* ``'gm'``  -- generative-model recovery (Bora et al., 2017): ``xhat_i = G(z_i)``
  with ``z_i = argmin_z sum_j w_j (a_j^T (x_i - G(z)))^2``.

The classical greedy algorithm is intractable here since it would reconstruct
the whole data set for every dictionary element at every enlargement; the
bilevel selection instead needs one implicit gradient per step.  Two baselines
from the paper are included: random RIP-style measurements and ``approx-greedy``
(Krause and Cevher, 2010), which picks the measurements with the largest average
inner product with the signals.
"""

import numpy as np
import torch
from torch.autograd import grad


def _cg_solve(matvec, b, max_iter=100, tol=1e-10):
    """Conjugate gradients for a symmetric positive definite ``matvec``."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.sum(r * r)
    target = tol * rs_old
    for _ in range(max_iter):
        if rs_old <= target or rs_old == 0:
            break
        ap = matvec(p)
        denom = torch.sum(p * ap)
        if torch.abs(denom) < 1e-20:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.sum(r * r)
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


class DictionarySelector(object):
    """Bilevel selection of compressed-sensing measurements (Eq. (12)).

    Args:
        A (torch.Tensor): dictionary of shape ``(m, d)``; row ``j`` is ``a_j``.
        recovery (str): ``'l2'``, ``'l1'`` or ``'gm'``.
        lam (float): ``lambda`` of Eq. (12); the paper uses ``0.01``.
        generator (torch.nn.Module): required for ``recovery='gm'``; maps
            latent codes ``(n, p)`` to signals ``(n, d)``.
        latent_dim (int): ``p``, required for ``recovery='gm'``.
        ista_iters (int), ista_lr (float): ISTA settings for ``recovery='l1'``.
        gm_iters (int), gm_lr (float): latent optimization settings for ``'gm'``.
        damping (float): ridge added to the inner Hessian before the CG solve;
            necessary because ``A_S^T A_S`` is rank deficient for small ``|S|``
            and because the ``L1`` penalty has zero curvature.
        cg_iters (int): CG iterations for the inverse-Hessian-vector product.
        device (str): torch device.
    """

    def __init__(self, A, recovery='l2', lam=0.01, generator=None, latent_dim=None,
                 ista_iters=200, ista_lr=None, gm_iters=300, gm_lr=0.05,
                 damping=1e-4, cg_iters=100, device='cpu', verbose=True, logging_period=1):
        self.A = A.to(device).float()
        self.recovery = recovery
        self.lam = lam
        self.generator = generator.to(device) if generator is not None else None
        self.latent_dim = latent_dim
        self.ista_iters = ista_iters
        self.ista_lr = ista_lr
        self.gm_iters = gm_iters
        self.gm_lr = gm_lr
        self.damping = damping
        self.cg_iters = cg_iters
        self.device = device
        self.verbose = verbose
        self.logging_period = logging_period
        if recovery == 'gm' and (generator is None or latent_dim is None):
            raise ValueError("recovery='gm' requires a generator and latent_dim")

    # ------------------------------------------------------------------
    # inner problem
    # ------------------------------------------------------------------
    def _gram(self, w):
        """``M = A^T D(w) A``."""
        return self.A.t() @ (w.unsqueeze(1) * self.A)

    def reconstruct(self, X, w):
        """Solve the inner problem of Eq. (12) for every signal.

        Returns:
            tuple: ``(Xhat, latent)`` where ``latent`` is the optimized latent
            code for ``recovery='gm'`` and ``None`` otherwise.
        """
        X = X.to(self.device).float()
        if self.recovery == 'l2':
            m = self._gram(w)
            d = m.shape[0]
            eye = torch.eye(d, device=self.device)
            # argmin_y (x - y)^T M (x - y) + lam ||y||^2  =>  (M + lam I) y = M x
            sol = torch.linalg.solve(m + self.lam * eye, (m @ X.t()))
            return sol.t(), None
        if self.recovery == 'l1':
            return self._ista(X, w), None
        return self._solve_gm(X, w)

    def _ista(self, X, w):
        """ISTA for ``min_y (x-y)^T M (x-y) + lam ||y||_1``."""
        m = self._gram(w)
        lip = 2.0 * torch.linalg.eigvalsh(m).max().clamp_min(1e-8)
        step = (1.0 / lip) if self.ista_lr is None else self.ista_lr
        y = X.clone()
        for _ in range(self.ista_iters):
            resid = (X - y) @ m            # (n, d)
            grad_y = -2.0 * resid
            y = y - step * grad_y
            thresh = step * self.lam
            y = torch.sign(y) * torch.clamp(torch.abs(y) - thresh, min=0.0)
        return y

    def _solve_gm(self, X, w):
        """Latent optimization ``z_i = argmin_z sum_j w_j (a_j^T (x_i - G(z)))^2``."""
        n = X.shape[0]
        z = torch.zeros(n, self.latent_dim, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=self.gm_lr)
        for _ in range(self.gm_iters):
            optimizer.zero_grad()
            loss = self._measurement_loss(X, self.generator(z), w).sum()
            loss.backward()
            optimizer.step()
        z = z.detach()
        return self.generator(z).detach(), z

    def _measurement_loss(self, X, Y, w):
        """``sum_j w_j (a_j^T (x_i - y_i))^2`` for every ``i``, shape ``(n,)``."""
        resid = (X - Y) @ self.A.t()       # (n, m)
        return torch.sum(w.unsqueeze(0) * resid ** 2, dim=1)

    # ------------------------------------------------------------------
    # implicit gradient
    # ------------------------------------------------------------------
    def implicit_grads(self, X, w):
        """``dG/dw_j`` of Eq. (12) for every dictionary element ``j``.

        The inner problems decouple over signals, so the Hessian is block
        diagonal and one conjugate-gradient solve over the stacked variable
        handles all blocks at once.  The mixed partial
        ``d^2 f / d var d w_j`` is obtained with a single double backward.
        """
        X = X.to(self.device).float()
        n = X.shape[0]
        w_var = w.detach().clone().requires_grad_(True)

        Xhat, z = self.reconstruct(X, w.detach())
        if self.recovery == 'gm':
            var = z.detach().clone().requires_grad_(True)

            def inner(v, weights):
                return self._measurement_loss(X, self.generator(v), weights).sum()

            def outer(v):
                return torch.sum((X - self.generator(v)) ** 2) / n
        else:
            var = Xhat.detach().clone().requires_grad_(True)
            reg = (lambda v: self.lam * torch.sum(v ** 2)) if self.recovery == 'l2' \
                else (lambda v: self.lam * torch.sum(torch.abs(v)))

            def inner(v, weights):
                return self._measurement_loss(X, v, weights).sum() + reg(v)

            def outer(v):
                return torch.sum((X - v) ** 2) / n

        # dg/dvar
        outer_grad = grad(outer(var), var)[0].detach()

        # H^-1 dg/dvar with H the Hessian of the inner objective (block diagonal)
        def matvec(p):
            f = inner(var, w.detach())
            g = grad(f, var, create_graph=True)[0]
            hv = grad(g, var, grad_outputs=p, retain_graph=False)[0].detach()
            return hv + self.damping * p

        u = _cg_solve(matvec, outer_grad, max_iter=self.cg_iters)

        # -u^T d^2 f / d var d w_j
        f = inner(var, w_var)
        g = grad(f, var, create_graph=True)[0]
        scores = -grad(g, w_var, grad_outputs=u)[0].detach()
        return scores

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------
    def reconstruction_error(self, X, inds):
        """Mean squared reconstruction error of Eq. (12) for a measurement set."""
        X = X.to(self.device).float()
        w = torch.zeros(self.A.shape[0], device=self.device)
        w[torch.as_tensor(np.asarray(inds), dtype=torch.long, device=self.device)] = 1.0
        Xhat, _ = self.reconstruct(X, w)
        return float(torch.mean(torch.sum((X - Xhat) ** 2, dim=1)).item())

    def select(self, X, k, selection_batch_size=1, init_inds=None):
        """Greedy bilevel forward selection of ``k`` measurements.

        Returns:
            np.ndarray: indices of the selected dictionary elements.
        """
        X = X.to(self.device).float()
        m = self.A.shape[0]
        selected = np.asarray([], dtype=int) if init_inds is None else np.asarray(init_inds, dtype=int)
        step = 0
        while len(selected) < k:
            w = torch.zeros(m, device=self.device)
            if len(selected) > 0:
                w[torch.as_tensor(selected, dtype=torch.long, device=self.device)] = 1.0
            if len(selected) == 0 and (self.recovery == 'gm' or self.lam <= 0):
                # cold start: with an empty support the inner problem of the
                # generative-model recovery is degenerate (zero Hessian), so
                # fall back to picking the measurement with the largest energy.
                # For the L2/L1 recovery this is unnecessary: at w = 0 the
                # Hessian is 2 * lam * I and the implicit gradient reduces
                # exactly to the same energy criterion.
                scores = -torch.sum((X @ self.A.t()) ** 2, dim=0)
            else:
                scores = self.implicit_grads(X, w)
            scores = scores.cpu().numpy()
            scores[selected] = np.inf
            take = int(min(selection_batch_size, k - len(selected)))
            chosen = np.argsort(scores)[:take]
            selected = np.concatenate([selected, chosen]).astype(int)
            step += 1
            if self.verbose and step % self.logging_period == 0:
                print('[dictionary] selected {}/{}, error {:.5f}'.format(
                    len(selected), k, self.reconstruction_error(X, selected)))
        return selected

    # ------------------------------------------------------------------
    # baselines
    # ------------------------------------------------------------------
    def select_random(self, k, rs=None):
        """Uniformly random measurements (RIP with high probability)."""
        rs = np.random if rs is None else rs
        return rs.choice(self.A.shape[0], k, replace=False)

    def select_approx_greedy(self, X, k):
        """Approximate greedy of Krause and Cevher (2010).

        Picks the measurements with the largest average inner product between
        the signals and the measurement vectors.
        """
        X = X.to(self.device).float()
        scores = torch.mean(torch.abs(X @ self.A.t()), dim=0)
        return torch.argsort(scores, descending=True)[:k].cpu().numpy()
