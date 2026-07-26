"""Coresets for Gaussian mixture models -- the unsupervised case (Sec. 5.2.1).

The bilevel formulation of Eq. (1) is not restricted to supervised learning: it
only needs a twice differentiable loss.  For a mixture model the loss is the
negative marginal log-likelihood

    l_i(theta) = -log( sum_k pi_k N(x_i | mu_k, Sigma_k) ),
    theta = {pi_k, mu_k, Sigma_k}_{k=1..K}

and Figure 5 of the paper shows that the resulting coresets beat the
GMM-specific sensitivity construction of Lucic et al. (2017) by an order of
magnitude in relative NLL error.

Two pieces are provided:

* :class:`WeightedGMM` -- a weighted EM fit, used as the inner solver (the
  paper minimizes the weighted NLL "using the EM algorithm").
* :class:`TorchGMM` -- the same model with an unconstrained, twice
  differentiable parameterization (softmax mixture logits, Cholesky factors
  with log-diagonal) so that Hessian-vector products are available through
  autograd.
* :class:`GMMCoreset` -- one-by-one forward selection with binary weights and
  conjugate-gradient inverse-Hessian-vector products, exactly the setting of
  Sec. 5.2.1.
"""

import numpy as np
import torch
import torch.nn as nn

from bicoreset.direct import BilevelCoreset

_LOG_2PI = float(np.log(2.0 * np.pi))


class WeightedGMM(object):
    """Gaussian mixture fitted by weighted EM.

    Args:
        n_components (int): number of mixture components ``K``.
        reg_covar (float): ridge added to the covariance diagonals.  Coresets
            are small, so a mildly generous value keeps the fit non-degenerate.
        max_iter (int): EM iterations.
        tol (float): relative tolerance on the weighted log-likelihood.
        n_init (int): number of random restarts; the best weighted likelihood
            wins.  EM on a handful of points has many poor local optima, so
            restarts matter much more for coresets than for the full data.
        seed (int): seed of the k-means++ style initialization.
    """

    def __init__(self, n_components=5, reg_covar=1e-6, max_iter=100, tol=1e-6,
                 n_init=1, seed=None):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.rs = np.random.RandomState(seed)
        self.pi = None
        self.mu = None
        self.sigma = None

    def _init_params(self, X, weights):
        """k-means++ style seeding, with the sampling biased by the weights."""
        n, d = X.shape
        k = self.n_components
        probs = np.clip(weights, 0, None)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n) / n
        centers = [X[self.rs.choice(n, p=probs)]]
        dists = np.sum((X - centers[0]) ** 2, axis=1)
        for _ in range(1, k):
            p = dists * probs
            p = p / p.sum() if p.sum() > 0 else probs
            center = X[self.rs.choice(n, p=p)]
            centers.append(center)
            dists = np.minimum(dists, np.sum((X - center) ** 2, axis=1))
        self.mu = np.stack(centers)
        cov = np.cov(X.T, aweights=weights) if n > 1 else np.eye(d)
        cov = np.atleast_2d(cov) + self.reg_covar * np.eye(d)
        self.sigma = np.stack([cov] * k)
        self.pi = np.ones(k) / k

    def _log_gaussian(self, X):
        n, d = X.shape
        out = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            cov = self.sigma[k] + self.reg_covar * np.eye(d)
            chol = np.linalg.cholesky(cov)
            diff = np.linalg.solve(chol, (X - self.mu[k]).T).T
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            out[:, k] = -0.5 * (d * _LOG_2PI + log_det + np.sum(diff ** 2, axis=1))
        return out

    def log_prob(self, X):
        """``log p(x_i)`` for every row of ``X``."""
        log_comp = self._log_gaussian(X) + np.log(np.clip(self.pi, 1e-12, None))
        mx = log_comp.max(axis=1, keepdims=True)
        return (mx + np.log(np.exp(log_comp - mx).sum(axis=1, keepdims=True))).ravel()

    def nll(self, X, weights=None):
        """Weighted negative log-likelihood ``-sum_i w_i log p(x_i)``."""
        lp = self.log_prob(X)
        if weights is None:
            return float(-lp.sum())
        return float(-(weights * lp).sum())

    def fit(self, X, weights=None):
        """Weighted EM with ``n_init`` restarts, keeping the best likelihood."""
        X = np.asarray(X, dtype=np.float64)
        weights = np.ones(X.shape[0]) if weights is None \
            else np.asarray(weights, dtype=np.float64)
        best = None
        for _ in range(max(1, self.n_init)):
            ll = self._fit_once(X, weights)
            if best is None or ll > best[0]:
                best = (ll, self.pi.copy(), self.mu.copy(), self.sigma.copy())
        _, self.pi, self.mu, self.sigma = best
        return self

    def _fit_once(self, X, weights):
        n, d = X.shape
        self._init_params(X, weights)
        prev = None
        ll = -np.inf
        for _ in range(self.max_iter):
            # E step
            log_comp = self._log_gaussian(X) + np.log(np.clip(self.pi, 1e-12, None))
            mx = log_comp.max(axis=1, keepdims=True)
            log_norm = mx + np.log(np.exp(log_comp - mx).sum(axis=1, keepdims=True))
            resp = np.exp(log_comp - log_norm)
            ll = float((weights * log_norm.ravel()).sum())

            # M step (responsibilities scaled by the sample weights)
            wr = resp * weights[:, None]
            nk = wr.sum(axis=0) + 1e-12
            self.pi = nk / nk.sum()
            self.mu = (wr.T @ X) / nk[:, None]
            for k in range(self.n_components):
                diff = X - self.mu[k]
                self.sigma[k] = (wr[:, k, None] * diff).T @ diff / nk[k]
                self.sigma[k] += self.reg_covar * np.eye(d)

            if prev is not None and abs(ll - prev) <= self.tol * max(1.0, abs(prev)):
                break
            prev = ll
        return ll


class TorchGMM(nn.Module):
    """Differentiable GMM; ``forward(X)`` returns ``log p(x_i)`` per sample.

    The parameterization is unconstrained -- mixture logits and Cholesky factors
    with a log-parameterized diagonal -- so that the NLL is twice differentiable
    in the parameters and Hessian-vector products are available.
    """

    def __init__(self, n_components, dim, reg_covar=1e-6):
        super(TorchGMM, self).__init__()
        self.n_components = n_components
        self.dim = dim
        self.reg_covar = reg_covar
        self.logits = nn.Parameter(torch.zeros(n_components))
        self.mu = nn.Parameter(torch.zeros(n_components, dim))
        self.chol_diag = nn.Parameter(torch.zeros(n_components, dim))
        self.chol_low = nn.Parameter(torch.zeros(n_components, dim, dim))

    def _chol(self):
        low = torch.tril(self.chol_low, diagonal=-1)
        return low + torch.diag_embed(torch.exp(self.chol_diag))

    def forward(self, X):
        chol = self._chol()
        log_pi = torch.log_softmax(self.logits, dim=0)
        comps = []
        for k in range(self.n_components):
            diff = (X - self.mu[k]).t()
            sol = torch.linalg.solve_triangular(chol[k], diff, upper=False)
            log_det = 2.0 * torch.sum(self.chol_diag[k])
            quad = torch.sum(sol ** 2, dim=0)
            comps.append(-0.5 * (self.dim * _LOG_2PI + log_det + quad) + log_pi[k])
        return torch.logsumexp(torch.stack(comps, dim=1), dim=1)

    @torch.no_grad()
    def load_from_em(self, gmm):
        """Copy the parameters of a fitted :class:`WeightedGMM`."""
        pi = torch.as_tensor(gmm.pi, dtype=torch.float32)
        self.logits.copy_(torch.log(torch.clamp(pi, min=1e-12)))
        self.mu.copy_(torch.as_tensor(gmm.mu, dtype=torch.float32))
        sigma = torch.as_tensor(gmm.sigma, dtype=torch.float32)
        eye = torch.eye(self.dim) * self.reg_covar
        chol = torch.linalg.cholesky(sigma + eye)
        diag = torch.diagonal(chol, dim1=-2, dim2=-1)
        self.chol_diag.copy_(torch.log(torch.clamp(diag, min=1e-8)))
        self.chol_low.copy_(torch.tril(chol, diagonal=-1))
        return self


def nll_loss(log_probs, _targets=None):
    """Per-sample negative log-likelihood; the ``targets`` argument is ignored."""
    return -log_probs


class GMMCoreset(object):
    """Bilevel coreset for a Gaussian mixture model (Sec. 5.2.1).

    The inner problem is solved by weighted EM, the implicit gradient is taken
    through the differentiable :class:`TorchGMM` parameterization with
    conjugate-gradient inverse-Hessian-vector products, and points are added
    one by one with binary weights -- the exact configuration described in the
    paper for Figures 4 and 5.

    Args:
        n_components (int): number of mixture components.
        reg_covar (float): covariance ridge, also acts as damping.
        em_iters (int): EM iterations per inner solve.
        cg_iters (int): CG iterations for the inverse-Hessian-vector product.
        damping (float): ridge on the Hessian; the NLL of a mixture is not
            convex, so some damping is needed for CG to behave.
        seed (int): random seed.
    """

    def __init__(self, n_components=5, reg_covar=1e-3, em_iters=100, em_restarts=5,
                 cg_iters=50, damping=1e-2, seed=None, device='cpu', verbose=True,
                 logging_period=5):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.em_iters = em_iters
        self.em_restarts = em_restarts
        self.cg_iters = cg_iters
        self.damping = damping
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.logging_period = logging_period
        self.coreset_builder = None

    def _make_builder(self, dim):
        def model_fn():
            return TorchGMM(self.n_components, dim, self.reg_covar)

        def train_fn(model, X, y, weights):
            gmm = WeightedGMM(self.n_components, reg_covar=self.reg_covar,
                              max_iter=self.em_iters, n_init=self.em_restarts,
                              seed=self.seed)
            gmm.fit(X.detach().cpu().numpy(), weights.detach().cpu().numpy())
            model.load_from_em(gmm)

        return BilevelCoreset(
            model_fn=model_fn,
            loss_fn=nll_loss,
            train_fn=train_fn,
            ihvp='cg',
            ihvp_kwargs={'max_iter': self.cg_iters, 'damping': self.damping},
            max_outer_it=0,
            retrain_from_scratch=True,
            device=self.device,
            verbose=self.verbose,
            logging_period=self.logging_period)

    def build(self, X, m, start_size=10, selection_batch_size=1, X_outer=None):
        """Select a coreset of ``m`` points for the mixture model.

        Args:
            X (np.ndarray or torch.Tensor): data of shape ``(n, d)``.
            m (int): coreset size.
            start_size (int): random initial subset (10 in the paper).
            selection_batch_size (int): points added per step.
            X_outer: data defining the outer objective; defaults to ``X``.

        Returns:
            tuple: ``(coreset_inds, coreset_weights)``.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.float()
        dummy_y = torch.zeros(X.shape[0])
        self.coreset_builder = self._make_builder(X.shape[1])
        outer = None if X_outer is None else torch.as_tensor(X_outer).float()
        outer_y = None if outer is None else torch.zeros(outer.shape[0])
        return self.coreset_builder.build(
            X, dummy_y, m,
            X_outer=outer, y_outer=outer_y,
            strategy='forward',
            selection_batch_size=selection_batch_size,
            start_size=start_size)

    @staticmethod
    def relative_nll_error(X, inds, n_components=5, weights=None, seed=None,
                           reg_covar=1e-3, n_init=5):
        """Relative NLL error of a subset fit w.r.t. the full-data fit (Figure 5)."""
        X = np.asarray(X, dtype=np.float64)
        full = WeightedGMM(n_components, reg_covar=reg_covar, n_init=n_init, seed=seed).fit(X)
        sub = WeightedGMM(n_components, reg_covar=reg_covar, n_init=n_init,
                          seed=seed).fit(X[inds], weights)
        full_nll = full.nll(X)
        return abs(sub.nll(X) - full_nll) / abs(full_nll)
