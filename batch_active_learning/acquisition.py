"""Acquisition strategies for semi-supervised batch active learning (Sec. 5.5).

All strategies share the signature

    ``fn(model, trainer, X_labeled, y_labeled, X_unlabeled, batch_size, **kwargs)
      -> np.ndarray``

returning indices *into* ``X_unlabeled``.  Implemented, matching Figure 10:

* :func:`uniform`      -- uniform subsampling;
* :func:`max_entropy`  -- highest predictive entropy (averaged over two
  augmentations);
* :func:`kcenter`      -- greedy k-center in the last-layer embedding
  (Sener and Savarese, 2018);
* :func:`consistency`  -- sum of per-class prediction variances under random
  augmentations (Gao et al., 2019);
* :func:`badge`        -- k-means++ on last-layer gradient embeddings of the
  hard pseudo-labels (Ash et al., 2020);
* :func:`bico`         -- the paper's proposal: build the coreset of the
  pseudo-labeled unlabeled pool, Eq. (10).
"""

import numpy as np
import torch
import torch.nn.functional as F

from bicoreset.direct import BilevelCoreset
from bicoreset.losses import soft_cross_entropy


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _probs(model, trainer, X, n_augmentations=1):
    return trainer.predict_proba(model, X, n_augmentations=n_augmentations).numpy()


@torch.no_grad()
def _embed(model, X, device='cpu', batch_size=512):
    """Last-layer embeddings; falls back to the logits if ``embed`` is missing."""
    model = model.to(device).eval()
    fn = getattr(model, 'embed', model)
    out = []
    for start in range(0, X.shape[0], batch_size):
        feat = fn(X[start:start + batch_size].to(device))
        out.append(feat.reshape(feat.shape[0], -1).cpu())
    return torch.cat(out).numpy()


def _kmeans_pp(X, k, rs=None):
    """k-means++ seeding; returns the indices of the chosen centers."""
    rs = np.random if rs is None else rs
    n = X.shape[0]
    k = min(k, n)
    inds = np.zeros(k, dtype=int)
    inds[0] = rs.choice(n)
    dists = np.sum((X - X[inds[0]]) ** 2, axis=1)
    for i in range(1, k):
        total = dists.sum()
        p = np.ones(n) / n if total <= 0 else dists / total
        ind = rs.choice(n, p=p)
        inds[i] = ind
        dists = np.minimum(dists, np.sum((X - X[ind]) ** 2, axis=1))
    return inds


# ----------------------------------------------------------------------
# strategies
# ----------------------------------------------------------------------
def uniform(model, trainer, X_l, y_l, X_u, batch_size, rs=None, **kwargs):
    rs = np.random if rs is None else rs
    return rs.choice(X_u.shape[0], batch_size, replace=False)


def max_entropy(model, trainer, X_l, y_l, X_u, batch_size, n_augmentations=2, **kwargs):
    p = _probs(model, trainer, X_u, n_augmentations=n_augmentations)
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)
    return np.argsort(-entropy)[:batch_size]


def kcenter(model, trainer, X_l, y_l, X_u, batch_size, device='cpu', **kwargs):
    """Greedy k-center in the embedding space, seeded with the labeled pool."""
    emb_l = _embed(model, X_l, device)
    emb_u = _embed(model, X_u, device)
    if emb_l.shape[0] > 0:
        sq = (emb_u ** 2).sum(1)[:, None] + (emb_l ** 2).sum(1)[None, :] - 2.0 * emb_u @ emb_l.T
        dists = np.maximum(sq, 0.0).min(axis=1)
    else:
        dists = np.full(emb_u.shape[0], np.inf)
    chosen = []
    for _ in range(batch_size):
        ind = int(np.argmax(dists))
        chosen.append(ind)
        dists = np.minimum(dists, np.sum((emb_u - emb_u[ind]) ** 2, axis=1))
        dists[ind] = -np.inf
    return np.asarray(chosen)


def consistency(model, trainer, X_l, y_l, X_u, batch_size, n_augmentations=5, **kwargs):
    """Gao et al. (2019): sum of per-class variances over random augmentations."""
    preds = []
    for _ in range(n_augmentations):
        preds.append(trainer.predict_proba(model, X_u, n_augmentations=2).numpy())
    preds = np.stack(preds)                     # (A, n, c)
    score = preds.var(axis=0).sum(axis=1)
    return np.argsort(-score)[:batch_size]


def badge(model, trainer, X_l, y_l, X_u, batch_size, device='cpu', rs=None,
          max_embedding_dim=200000, **kwargs):
    """BADGE (Ash et al., 2020): k-means++ on last-layer gradient embeddings."""
    emb = _embed(model, X_u, device)
    p = _probs(model, trainer, X_u)
    hard = np.argmax(p, axis=1)
    n, e = emb.shape
    c = p.shape[1]
    if n * c * e > max_embedding_dim * 100:
        raise MemoryError('BADGE gradient embedding too large ({} x {})'.format(n, c * e))
    scale = p.copy()
    scale[np.arange(n), hard] -= 1.0
    grad_emb = (scale[:, :, None] * emb[:, None, :]).reshape(n, c * e)
    return _kmeans_pp(grad_emb, batch_size, rs)


def bico(model, trainer, X_l, y_l, X_u, batch_size,
         feature_fn=None, num_classes=None, inner_reg=1e-4,
         max_inner_it=500, inner_lr=1e-2, cg_iters=100,
         device='cpu', verbose=False, presampled_features=None, **kwargs):
    """Bilevel coreset acquisition, Eq. (10).

    Selects ``M`` such that ``D_train u M`` is the coreset of
    ``D_train u D_u_hat``, where ``D_u_hat`` are the unlabeled points carrying
    the soft pseudo-labels of the semi-supervised learner:

        M = argmin_{M subset D_u_hat, |M| = m}
              sum_{D_train} l(h_theta*(x), y) + sum_{D_u_hat} l(h_theta*(x), yhat)
            s.t. theta* = argmin_theta sum_{D_train} l + sum_M l(., yhat)

    The inner problem is the convex proxy of Sec. 3.5.2: multiclass logistic
    regression on the (Nystrom) proxy features with an ``L2`` penalty, which
    makes it strongly convex.

    Args:
        feature_fn (callable): maps raw inputs to proxy features, e.g. a
            :class:`~batch_active_learning.proxy.NystromFeatureMap`.  Defaults
            to the model's own last-layer embedding.
        num_classes (int): number of classes; inferred from the model output
            when omitted.
        presampled_features (tuple): optional ``(Z_l, Z_u)`` precomputed proxy
            features, to avoid recomputing the kernel every round.
    """
    import models as target_models

    probs_u = trainer.predict_proba(model, X_u).float()
    if num_classes is None:
        num_classes = probs_u.shape[1]

    if presampled_features is not None:
        Z_l, Z_u = presampled_features
    elif feature_fn is not None:
        Z_l = torch.from_numpy(np.asarray(feature_fn(X_l.cpu().numpy()))).float()
        Z_u = torch.from_numpy(np.asarray(feature_fn(X_u.cpu().numpy()))).float()
    else:
        Z_l = torch.from_numpy(_embed(model, X_l, device)).float()
        Z_u = torch.from_numpy(_embed(model, X_u, device)).float()

    Y_l = F.one_hot(y_l.cpu().long(), num_classes).float()
    Z = torch.cat([Z_l, Z_u])
    Y = torch.cat([Y_l, probs_u])
    n_l = Z_l.shape[0]
    feature_dim = Z.shape[1]

    def model_fn():
        return target_models.LogisticRegression(feature_dim, num_classes)

    builder = BilevelCoreset(
        model_fn=model_fn,
        loss_fn=soft_cross_entropy,
        inner_reg=inner_reg,
        ihvp='cg',
        ihvp_kwargs={'max_iter': cg_iters, 'damping': inner_reg},
        max_inner_it=max_inner_it,
        inner_lr=inner_lr,
        max_outer_it=0,
        retrain_from_scratch=True,
        device=device,
        verbose=verbose)

    inds, _ = builder.build(
        Z, Y, batch_size,
        X_outer=Z, y_outer=Y,
        base_inds=np.arange(n_l),
        selectable_inds=np.arange(n_l, Z.shape[0]),
        strategy='forward',
        selection_batch_size=1,
        start_size=0)
    selected = np.asarray([i for i in inds if i >= n_l], dtype=int) - n_l
    return selected


ACQUISITION_FNS = {
    'uniform': uniform,
    'max_entropy': max_entropy,
    'kcenter': kcenter,
    'consistency': consistency,
    'badge': badge,
    'bico': bico,
}


def get_acquisition_fn(name):
    if name not in ACQUISITION_FNS:
        raise ValueError('unknown acquisition "{}", available: {}'.format(
            name, sorted(ACQUISITION_FNS)))
    return ACQUISITION_FNS[name]
