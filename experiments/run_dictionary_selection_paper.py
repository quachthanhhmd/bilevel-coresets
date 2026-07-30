"""Literal reproduction of Sec. 5.6 (Figure 12): dictionary selection for
compressed sensing.

Paper setup, quoted verbatim (Sec. 5.6; no appendix cross-reference for this
section -- everything needed is in the main text):

* "a synthetic data set containing a set of random sparse vectors, and the
  recovery of MNIST digits using a variational autoencoder (VAE) [...]
  coupled with the reconstruction method of Bora et al. (2017)."
* Synthetic: "contains 1024 vectors in 128-dimensional Euclidean space where
  the sparsity level is set to 10%, meaning only approximately 12 values are
  nonzero per vector. [...] the dictionary elements (measurements) are
  vectors in 128 dimensional space that have normally distributed entries
  with mean zero and variance 1/128."
* MNIST: "has dimensionality 28^2 [...] we select 250 at random for
  computational efficiency."
* "In both data sets, we used lambda = 0.01 as in Eq. (12)."
* Baselines: "randomly sampled measurements" and "approx-greedy [...]
  inspired by the heuristics of Krause and Cevher (2010)".
* Dictionary types: "a set of random matrices with entries distributed
  according to the unit normal distribution, or a wavelets basis (db1
  wavelet) for MNIST as done by Bora et al. (2017)."
* VAE: "the architecture and loss function [...] as in Kingma and Welling
  (2014) where we chose the latent vectors to be 20-dimensional."
* Figure 12 caption: "The size of the dictionary is 16384, 786, and 786 from
  left to right" -- i.e. (1) synthetic/Gaussian dictionary: 16384 candidate
  measurements; (2) MNIST/wavelet dictionary, L1 and L2 recovery: 786
  candidates; (3) MNIST/VAE, generative-model (GM) recovery: 786 candidates.
  Our db1 dictionary comes out to 841 candidates under the (unspecified by
  the paper) 'periodization' boundary convention -- see
  ``wavelet_dictionary_db1`` below.

This reproduces all three panels of Figure 12 using
``bicoreset.dictionary.DictionarySelector`` (already implements 'l2', 'l1'
and 'gm' recovery per Eq. (12), plus the 'random' and 'approx-greedy'
baselines) -- no new selection math is needed here, only the exact paper
data/dictionary setup and, for panel 3, a small VAE trained to the
Kingma & Welling (2014) architecture/loss.

Compute cost: panel 1 (m=16384 candidates, d=128, n=1024 signals) is the
most expensive -- each greedy step costs one CG solve over all live signals;
picking many measurements one at a time is slow, so ``--selection-batch``
lets you add several measurements per step (the paper does not state a
batch size for this experiment; batches trade off fidelity to a literal
one-by-one forward selection for tractability). Panels 2/3 (786 candidates,
250 signals) are cheap. Training the VAE for panel 3 is a few minutes on
CPU for the given dimensionality.

Run with::

    python experiments/run_dictionary_selection_paper.py
    python experiments/run_dictionary_selection_paper.py --panels synthetic --sizes 4,16,64,256
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bicoreset.dictionary import DictionarySelector


# ----------------------------------------------------------------------
# panel 1: synthetic sparse vectors + Gaussian dictionary
# ----------------------------------------------------------------------
def synthetic_sparse_signals(n=1024, d=128, sparsity_frac=0.10, seed=0):
    """n signals in R^d, ~sparsity_frac * d nonzero entries each (paper: ~12/128)."""
    rs = np.random.RandomState(seed)
    sparsity = max(1, int(round(sparsity_frac * d)))
    X = np.zeros((n, d))
    for i in range(n):
        support = rs.choice(d, sparsity, replace=False)
        X[i, support] = rs.randn(sparsity)
    return torch.from_numpy(X).float()


def gaussian_dictionary(m, d, seed=1):
    """m measurement vectors in R^d, entries ~ N(0, 1/d) (paper: variance 1/128)."""
    return torch.from_numpy(np.random.RandomState(seed).randn(m, d) / np.sqrt(d)).float()


# ----------------------------------------------------------------------
# panel 2/3: MNIST + wavelet dictionary / VAE generative recovery
# ----------------------------------------------------------------------
def load_mnist_subset(n=250, seed=0, data_root='data'):
    from torchvision import datasets, transforms
    ds = datasets.MNIST(data_root, train=True, download=True, transform=transforms.ToTensor())
    rs = np.random.RandomState(seed)
    inds = rs.choice(len(ds), n, replace=False)
    X = torch.stack([ds[i][0].reshape(-1) for i in inds])  # (n, 784)
    return X


def wavelet_dictionary_db1(d=784):
    """db1 (Haar) wavelet basis for a flattened 28x28 image, via pywavelets.

    A full db1 decomposition of a 28x28 image over all levels yields the
    same number of coefficients as pixels padded to the wavelet's supported
    size; we build the basis by taking the wavelet transform of each
    standard basis vector (columns of the inverse transform), which is
    exactly the "wavelets basis (db1 wavelet)" dictionary construction used
    by Bora et al. (2017)/the paper. The exact dictionary size depends on
    the boundary/padding convention used for the transform, which the paper
    does not specify; with ``pywt``'s 'periodization' mode on a 28x28 image
    this comes out to 841 (29x29), close to but not exactly the paper's
    stated 786 -- treat this as a known, undocumented-by-the-paper deviation
    rather than a bug.
    """
    import pywt

    side = int(round(d ** 0.5))
    probe = pywt.wavedec2(np.zeros((side, side)), 'db1', mode='periodization')
    coeffs, slices = pywt.coeffs_to_array(probe)
    m = coeffs.size
    basis = np.zeros((m, d), dtype=np.float32)
    flat = np.zeros(m)
    for j in range(m):
        flat[:] = 0.0
        flat[j] = 1.0
        arr = flat.reshape(coeffs.shape)
        rec_coeffs = pywt.array_to_coeffs(arr, slices, output_format='wavedec2')
        img = pywt.waverec2(rec_coeffs, 'db1', mode='periodization')
        basis[j] = img[:side, :side].reshape(-1)
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    basis = basis / norms
    return torch.from_numpy(basis).float()


class VAE(nn.Module):
    """Kingma & Welling (2014) MLP VAE, latent_dim=20 (paper's choice for MNIST)."""

    def __init__(self, input_dim=784, hidden=400, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_fc1 = nn.Linear(input_dim, hidden)
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_logvar = nn.Linear(hidden, latent_dim)
        self.dec_fc1 = nn.Linear(latent_dim, hidden)
        self.dec_out = nn.Linear(hidden, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, z):
        """Generator interface expected by DictionarySelector(recovery='gm')."""
        return self.decode(z)


def train_vae(X, latent_dim=20, epochs=30, batch_size=128, lr=1e-3, seed=0, verbose=True):
    torch.manual_seed(seed)
    model = VAE(input_dim=X.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X[idx]
            mu, logvar = model.encode(xb)
            z = model.reparameterize(mu, logvar)
            xhat = model.decode(z)
            recon = F.binary_cross_entropy(xhat, xb, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print('[vae] epoch {}/{} loss {:.1f}'.format(epoch + 1, epochs, total / n))
    return model.eval()


# ----------------------------------------------------------------------
# generic comparison routine
# ----------------------------------------------------------------------
def compare(name, X, A, sizes, recovery, lam, selection_batch, seed, extra_kwargs=None):
    kwargs = dict(recovery=recovery, lam=lam, verbose=False)
    kwargs.update(extra_kwargs or {})
    selector = DictionarySelector(A, **kwargs)
    print('\n=== {} (recovery={}, |A|={}) ==='.format(name, recovery.upper(), A.shape[0]))
    print('{:>6} {:>14} {:>16} {:>14}'.format('k', 'random', 'approx-greedy', 'bilevel'))
    rows = []
    selected = None
    for k in sizes:
        rs = np.random.RandomState(seed)
        rand = float(np.mean([selector.reconstruction_error(X, selector.select_random(k, rs))
                              for _ in range(3)]))
        greedy = selector.reconstruction_error(X, selector.select_approx_greedy(X, k))
        selected = selector.select(X, k, selection_batch_size=selection_batch,
                                   init_inds=selected)
        bilevel = selector.reconstruction_error(X, selected)
        print('{:>6} {:>14.5f} {:>16.5f} {:>14.5f}'.format(k, rand, greedy, bilevel))
        rows.append(dict(k=k, random=rand, approx_greedy=greedy, bilevel=bilevel))
    return rows


def plot_panel(rows, title, out_path):
    import matplotlib.pyplot as plt
    ks = [r['k'] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for key, label, marker in [('random', 'Random', 'o'), ('approx_greedy', 'Approx-greedy', 's'),
                               ('bilevel', 'Bilevel', '^')]:
        ax.plot(ks, [r[key] for r in rows], marker=marker, label=label)
    ax.set_xlabel('Subset Size (measurements)')
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print('saved {}'.format(out_path))


def main():
    parser = argparse.ArgumentParser(description='Sec. 5.6 dictionary selection reproduction (Figure 12)')
    parser.add_argument('--panels', default='synthetic,mnist_wavelet,mnist_vae',
                        help='comma-separated subset of synthetic,mnist_wavelet,mnist_vae')
    parser.add_argument('--lam', type=float, default=0.01, help='paper: lambda = 0.01 in both data sets')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', default=None)

    # synthetic panel (paper defaults)
    parser.add_argument('--synth-n', type=int, default=1024, help='paper: 1024 signals')
    parser.add_argument('--synth-d', type=int, default=128, help='paper: 128-dimensional')
    parser.add_argument('--synth-sparsity-frac', type=float, default=0.10, help='paper: 10% sparsity')
    parser.add_argument('--synth-dict-size', type=int, default=16384, help='paper: 16384 candidate measurements')
    parser.add_argument('--synth-sizes', default='16,64,256,1024',
                        help='coreset sizes for the synthetic panel (not specified by the paper)')
    parser.add_argument('--synth-selection-batch', type=int, default=8,
                        help='measurements added per greedy step for the 16384-candidate dictionary '
                             '(paper does not state a batch size for this panel; one-by-one is very '
                             'slow at this dictionary size, so we batch by default -- pass 1 for a '
                             'literal one-by-one forward selection)')

    # MNIST panels (paper defaults)
    parser.add_argument('--mnist-n', type=int, default=250, help='paper: 250 images selected at random')
    parser.add_argument('--mnist-sizes', default='16,64,256,786',
                        help='coreset sizes for the MNIST panels (paper dictionary size is 786)')
    parser.add_argument('--vae-latent-dim', type=int, default=20, help='paper: 20-dimensional latent')
    parser.add_argument('--vae-epochs', type=int, default=30)
    parser.add_argument('--vae-train-n', type=int, default=5000,
                        help='number of MNIST images used to train the VAE (separate from the 250 '
                             'signals being reconstructed; not specified by the paper)')

    args = parser.parse_args()
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    panels = set(args.panels.split(','))

    if 'synthetic' in panels:
        X = synthetic_sparse_signals(args.synth_n, args.synth_d, args.synth_sparsity_frac, seed=args.seed)
        A = gaussian_dictionary(args.synth_dict_size, args.synth_d, seed=args.seed + 1)
        sizes = [int(s) for s in args.synth_sizes.split(',')]
        for recovery in ('l2', 'l1'):
            extra = dict(ista_iters=100, damping=1e-2, cg_iters=50) if recovery == 'l1' else dict(damping=1e-4)
            rows = compare('Synthetic sparse vectors (Gaussian dictionary)', X, A, sizes, recovery,
                           args.lam, args.synth_selection_batch, args.seed, extra)
            plot_panel(rows, 'Synthetic, {} recovery'.format(recovery.upper()),
                      os.path.join(out_dir, 'dictionary_synthetic_{}.png'.format(recovery)))

    if 'mnist_wavelet' in panels or 'mnist_vae' in panels:
        X_mnist = load_mnist_subset(args.mnist_n, seed=args.seed)
        sizes = [int(s) for s in args.mnist_sizes.split(',')]

    if 'mnist_wavelet' in panels:
        try:
            A_wav = wavelet_dictionary_db1(X_mnist.shape[1])
        except ImportError:
            print('pywavelets not installed -- skipping mnist_wavelet panel '
                 '(pip install PyWavelets --break-system-packages)')
        else:
            for recovery in ('l2', 'l1'):
                extra = dict(ista_iters=100, damping=1e-2, cg_iters=50) if recovery == 'l1' else dict(damping=1e-4)
                rows = compare('MNIST (db1 wavelet dictionary)', X_mnist, A_wav, sizes, recovery,
                               args.lam, 1, args.seed, extra)
                plot_panel(rows, 'MNIST wavelet, {} recovery'.format(recovery.upper()),
                          os.path.join(out_dir, 'dictionary_mnist_wavelet_{}.png'.format(recovery)))

    if 'mnist_vae' in panels:
        X_vae_train = load_mnist_subset(args.vae_train_n, seed=args.seed + 100)
        print('\ntraining VAE (latent_dim={}) on {} MNIST images...'.format(
            args.vae_latent_dim, X_vae_train.shape[0]))
        vae = train_vae(X_vae_train, latent_dim=args.vae_latent_dim, epochs=args.vae_epochs, seed=args.seed)

        # measurement dictionary in the VAE's data space (paper: dictionary size 786,
        # same as the wavelet panel -- random Gaussian measurement vectors of the VAE's
        # *output* dimensionality, since GM recovery reconstructs in image space)
        A_gm = gaussian_dictionary(786, X_mnist.shape[1], seed=args.seed + 2)
        rows = compare('MNIST (VAE generative-model recovery, Bora et al. 2017)', X_mnist, A_gm, sizes, 'gm',
                       args.lam, 1, args.seed,
                       extra_kwargs=dict(generator=vae, latent_dim=args.vae_latent_dim,
                                         gm_iters=300, gm_lr=0.05, damping=1e-2, cg_iters=100))
        plot_panel(rows, 'MNIST VAE, generative-model recovery',
                  os.path.join(out_dir, 'dictionary_mnist_vae_gm.png'))


if __name__ == '__main__':
    main()
