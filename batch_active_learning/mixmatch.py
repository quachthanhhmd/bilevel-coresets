"""MixMatch (Berthelot et al., 2019), the SSL learner used in Sec. 5.5.

The batch active learning strategy of the paper is oblivious to the choice of
the semi-supervised algorithm -- "it only assumes that the semi-supervised
training outperforms supervised training of the model in terms of the
generalization error" -- but all experiments in Sec. 5.5 use MixMatch, so it is
implemented here to make the acquisition pipeline self-contained.

The implementation operates on plain tensors plus an ``augment_fn`` callable,
which keeps it usable for images (random crop/flip) as well as for the audio
spectrograms of the paper (amplitude / speed / shift / background-noise
augmentations).
"""

import numpy as np
import torch
import torch.nn.functional as F


def sharpen(p, temperature):
    """Sharpening of a distribution: ``p^(1/T)`` renormalized."""
    p = p ** (1.0 / temperature)
    return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)


def mixup(x1, p1, x2, p2, alpha):
    """MixUp with ``lambda' = max(lambda, 1 - lambda)`` as in MixMatch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)
    return lam * x1 + (1.0 - lam) * x2, lam * p1 + (1.0 - lam) * p2


def linear_rampup(step, rampup_steps):
    if rampup_steps <= 0:
        return 1.0
    return float(np.clip(step / rampup_steps, 0.0, 1.0))


class MixMatchTrainer(object):
    """Train a classifier with MixMatch.

    Args:
        num_classes (int): number of classes.
        augment_fn (callable): ``augment_fn(x_batch) -> x_batch``; identity if
            ``None``.
        n_augmentations (int): ``K`` augmentations used for label guessing
            (2 in the paper).
        temperature (float): sharpening temperature ``T``.
        alpha (float): MixUp Beta parameter.
        lambda_u (float): unlabeled loss weight (10 in the paper).
        epochs (int), batch_size (int), lr (float): optimization settings.
        rampup_epochs (int): linear ramp-up of the unlabeled loss weight.
        optimizer_fn (callable): ``optimizer_fn(params) -> optimizer``.
        scheduler_fn (callable): ``scheduler_fn(optimizer, epochs) -> scheduler``.
        device (str): torch device.
    """

    def __init__(self, num_classes, augment_fn=None, n_augmentations=2, temperature=0.5,
                 alpha=0.75, lambda_u=10.0, epochs=30, batch_size=64, lr=1e-3,
                 rampup_epochs=None, optimizer_fn=None, scheduler_fn=None,
                 device='cpu', verbose=False):
        self.num_classes = num_classes
        self.augment_fn = augment_fn if augment_fn is not None else (lambda x: x)
        self.n_augmentations = n_augmentations
        self.temperature = temperature
        self.alpha = alpha
        self.lambda_u = lambda_u
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.rampup_epochs = epochs if rampup_epochs is None else rampup_epochs
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.device = device
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _one_hot(self, y):
        return F.one_hot(y.long(), self.num_classes).float()

    def _guess_labels(self, model, xu):
        """Average the predictions over ``K`` augmentations and sharpen."""
        with torch.no_grad():
            probs = 0
            for _ in range(self.n_augmentations):
                probs = probs + torch.softmax(model(self.augment_fn(xu)), dim=1)
            probs = probs / self.n_augmentations
            return sharpen(probs, self.temperature)

    def train(self, model, X_l, y_l, X_u=None):
        """Run MixMatch (or plain supervised training when ``X_u`` is ``None``)."""
        model = model.to(self.device).train()
        params = model.parameters()
        optimizer = self.optimizer_fn(params) if self.optimizer_fn is not None \
            else torch.optim.Adam(params, lr=self.lr)
        scheduler = self.scheduler_fn(optimizer, self.epochs) if self.scheduler_fn is not None else None

        X_l = X_l.to(self.device)
        y_l = y_l.to(self.device)
        n_l = X_l.shape[0]
        has_unlabeled = X_u is not None and X_u.shape[0] > 0
        if has_unlabeled:
            X_u = X_u.to(self.device)
            n_u = X_u.shape[0]

        steps_per_epoch = max(1, int(np.ceil(n_l / self.batch_size)))
        for epoch in range(self.epochs):
            weight_u = self.lambda_u * linear_rampup(epoch, self.rampup_epochs)
            for _ in range(steps_per_epoch):
                idx_l = torch.randint(0, n_l, (min(self.batch_size, n_l),), device=self.device)
                xb = self.augment_fn(X_l[idx_l])
                pb = self._one_hot(y_l[idx_l])

                if has_unlabeled:
                    idx_u = torch.randint(0, n_u, (min(self.batch_size, n_u),), device=self.device)
                    xu = X_u[idx_u]
                    qb = self._guess_labels(model, xu)
                    xu_aug = torch.cat([self.augment_fn(xu) for _ in range(self.n_augmentations)])
                    qb_rep = qb.repeat(self.n_augmentations, 1)

                    all_x = torch.cat([xb, xu_aug])
                    all_p = torch.cat([pb, qb_rep])
                    perm = torch.randperm(all_x.shape[0], device=self.device)
                    mixed_x, mixed_p = mixup(all_x, all_p, all_x[perm], all_p[perm], self.alpha)

                    logits = model(mixed_x)
                    n_lab = xb.shape[0]
                    loss_x = -torch.mean(torch.sum(
                        torch.log_softmax(logits[:n_lab], dim=1) * mixed_p[:n_lab], dim=1))
                    loss_u = torch.mean((torch.softmax(logits[n_lab:], dim=1) - mixed_p[n_lab:]) ** 2)
                    loss = loss_x + weight_u * loss_u
                else:
                    perm = torch.randperm(xb.shape[0], device=self.device)
                    mixed_x, mixed_p = mixup(xb, pb, xb[perm], pb[perm], self.alpha)
                    logits = model(mixed_x)
                    loss = -torch.mean(torch.sum(torch.log_softmax(logits, dim=1) * mixed_p, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                print('[mixmatch] epoch {}/{} loss {:.4f}'.format(epoch + 1, self.epochs, float(loss)))
        return model

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, model, X, n_augmentations=1, batch_size=512):
        """Class probabilities, optionally averaged over augmentations."""
        model = model.to(self.device).eval()
        outputs = []
        for start in range(0, X.shape[0], batch_size):
            xb = X[start:start + batch_size].to(self.device)
            probs = 0
            for i in range(max(1, n_augmentations)):
                inp = self.augment_fn(xb) if n_augmentations > 1 else xb
                probs = probs + torch.softmax(model(inp), dim=1)
            outputs.append((probs / max(1, n_augmentations)).cpu())
        return torch.cat(outputs)

    @torch.no_grad()
    def accuracy(self, model, X, y, batch_size=512):
        probs = self.predict_proba(model, X, batch_size=batch_size)
        return float((torch.argmax(probs, dim=1) == y.cpu().long()).float().mean())
