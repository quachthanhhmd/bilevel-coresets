"""Per-sample loss functions used by the bilevel coreset constructions.

Throughout this package a loss function has the signature

    ``loss_fn(outputs, targets) -> torch.Tensor`` of shape ``(n,)``

i.e. it returns the *per-sample* losses, not a reduced scalar.  This makes the
inner objective ``f(theta, w) = sum_i w_i * l_i(theta)`` of Eq. (1) and the
mixed partial ``d^2 f / d theta d w_k = grad_theta l_k(theta)`` of Eq. (5)
expressible without ambiguity about the reduction.
"""

import torch
import torch.nn.functional as F


def cross_entropy(outputs, targets):
    """Multiclass cross entropy with hard integer labels.

    Args:
        outputs (torch.Tensor): logits of shape ``(n, c)``.
        targets (torch.Tensor): integer labels of shape ``(n,)``.

    Returns:
        torch.Tensor: per-sample losses of shape ``(n,)``.
    """
    return F.cross_entropy(outputs, targets.long(), reduction='none')


def soft_cross_entropy(outputs, targets):
    """Cross entropy against soft targets (probabilities), used for pseudo-labels.

    Args:
        outputs (torch.Tensor): logits of shape ``(n, c)``.
        targets (torch.Tensor): probabilities of shape ``(n, c)``; they are
            normalized internally so that unnormalized scores also work.

    Returns:
        torch.Tensor: per-sample losses of shape ``(n,)``.
    """
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -torch.sum(torch.log_softmax(outputs, dim=1) * targets, dim=1)


def logit_cross_entropy(outputs, targets):
    """Cross entropy where the targets are *logits* (softmaxed internally).

    Matches the convention of ``batch_active_learning/nystrom_example.py`` in
    the original repository.
    """
    return -torch.sum(torch.log_softmax(outputs, dim=1) * torch.softmax(targets, dim=1), dim=1)


def mse(outputs, targets):
    """Squared error summed over the output dimensions, per sample."""
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(1)
    return torch.sum((outputs - targets) ** 2, dim=1)


def accuracy(outputs, targets):
    """Classification accuracy; ``targets`` may be hard labels or probabilities."""
    if targets.dim() > 1:
        targets = torch.argmax(targets, dim=1)
    return (torch.argmax(outputs, dim=1) == targets.long()).float().mean()


def l2_penalty(model):
    """``||theta||_2^2`` over all parameters of ``model``."""
    return sum(torch.sum(p * p) for p in model.parameters())
