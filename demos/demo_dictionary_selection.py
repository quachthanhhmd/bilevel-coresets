"""Dictionary selection for compressed sensing (Sec. 4.5 / 5.6, Figure 12).

Reconstruction error against the number of selected measurements on synthetic
sparse signals, comparing random RIP-style measurements, the approximate-greedy
heuristic of Krause and Cevher (2010), and the bilevel selection of Eq. (12),
with both L2 and L1 recovery.

Two dictionaries are used, as in the paper: "a set of random matrices with
entries distributed according to the unit normal distribution, or a wavelets
basis [...] which is a more challenging baseline since not necessarily all
elements are equally sparse".  Here the structured dictionary is a DCT basis;
this is where informed selection pays off most.

Run with::

    python demos/demo_dictionary_selection.py
"""

import numpy as np
import torch

from _common import set_seed

from bicoreset.dictionary import DictionarySelector


def dct_basis(d):
    j = np.arange(d)
    basis = np.cos(np.pi * (j[:, None] + 0.5) * j[None, :] / d).T
    return basis / np.linalg.norm(basis, axis=1, keepdims=True)


def random_dictionary(m, d, seed=1):
    return np.random.RandomState(seed).randn(m, d) / np.sqrt(d)


def sparse_signals(n=200, d=64, sparsity=6, basis=None, active=None, seed=0):
    """Signals that are ``sparsity``-sparse, optionally in the basis ``basis``."""
    rs = np.random.RandomState(seed)
    coef = np.zeros((n, d))
    active = d if active is None else active
    for i in range(n):
        support = rs.choice(active, sparsity, replace=False)
        coef[i, support] = rs.randn(sparsity)
    X = coef if basis is None else coef @ basis
    return torch.from_numpy(X).float()


def compare(name, X, A, sizes):
    print('\n=== {} ==='.format(name))
    for recovery in ('l2', 'l1'):
        kwargs = dict(recovery=recovery, lam=0.01, verbose=False)
        if recovery == 'l1':
            kwargs.update(ista_iters=100, damping=1e-2, cg_iters=50)
        selector = DictionarySelector(A, **kwargs)
        print('recovery = {}'.format(recovery.upper()))
        print('{:>6} {:>12} {:>14} {:>12}'.format('k', 'random', 'approx-greedy', 'bilevel'))
        for k in sizes:
            rs = np.random.RandomState(0)
            rand = np.mean([selector.reconstruction_error(X, selector.select_random(k, rs))
                            for _ in range(3)])
            greedy = selector.reconstruction_error(X, selector.select_approx_greedy(X, k))
            bilevel = selector.reconstruction_error(X, selector.select(X, k))
            print('{:>6} {:>12.5f} {:>14.5f} {:>12.5f}'.format(k, rand, greedy, bilevel))


def main():
    set_seed(0)
    d = 64

    # (a) i.i.d. Gaussian dictionary, signals sparse in the canonical basis
    X = sparse_signals(n=200, d=d, sparsity=6)
    A = torch.from_numpy(random_dictionary(128, d)).float()
    compare('random Gaussian dictionary (RIP w.h.p.)', X, A, [8, 16, 32])

    # (b) structured DCT dictionary, signals sparse in the low/mid frequencies
    basis = dct_basis(d)
    X = sparse_signals(n=200, d=d, sparsity=6, basis=basis, active=24)
    A = torch.from_numpy(basis).float()
    compare('structured DCT dictionary', X, A, [4, 8, 16])


if __name__ == '__main__':
    main()
