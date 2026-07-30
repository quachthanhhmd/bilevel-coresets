"""Literal reproduction of Sec. 5.5 (Table 7, Figure 10): semi-supervised
batch active learning on audio keyword recognition.

Paper setup, quoted verbatim (Sec. 5.5; no Appendix C cross-reference found
for this section -- all hyperparameters below are from the main text):

* Data: "Spoken Digit data set (Jackson, 2016) (2700 utterances, 10 classes)
  and Speech Commands V2 (Warden, 2018) (85000 utterances, 35 classes)".
* Features: "we map the utterances to 32x32 mel spectrograms by first
  resampling them to 16kHz and applying the mel feature extraction with
  window length of 128 ms, hop length of 32 ms and 32 bins."
* Model: "Wide ResNet-28-10 (Zagoruyko and Komodakis, 2016) with weight
  decay of 10^-4 and without dropout."
* SSL: "MixMatch with two augmentations for label guessing and unlabeled
  cost weight lambda_u = 10, with other hyperparameters set to their
  defaults (Berthelot et al., 2019)."
* Audio augmentations, applied "in order with 0.5 probability" each:
  "i) amplitude change by a ~ U(0.8, 1.2), ii) audio speed change by
  s ~ U(0.8, 1.2), iii) random time shifts by t ms, where t ~ U(-250, 250),
  iv) mixing in background noise with SNR r dB, where r ~ U(0, 40)" (noise
  segments taken from the Speech Commands data set).
* Training: "Adam with an initial learning rate 10^-3 cosine annealed to 0
  over 30 epochs" (Table 7 supervised/SSL baselines).
* Acquisition (the paper's proposal): "we solve the coreset selection
  problem in Equation (10) with the CNTK proxy with cross-entropy loss
  (Section 3.5.2) with 2048-dimensional features and we add 10^-4 L2
  penalty to the inner objective [...] for each labeled point, we presample
  100 augmentations [...] and concatenate them for batch gradient descent.
  We perform one-by-one forward selection, with approximate implicit
  gradients obtained using 100 steps of conjugate gradients."
* Budget: "small labeled pools (n_labeled <= 200) and perform the
  acquisition in batches of size m = 10 starting with 10 and 50 labeled
  samples for Spoken Digit and Speech Commands, respectively." Starting
  pools "guaranteed to contain at least one sample from each class."
* "in every round of active learning, we retrain the models from scratch
  until convergence using MixMatch."
* Baselines: uniform, max-entropy (2-augmentation average), k-center on the
  last-layer embedding (Sener and Savarese, 2018), consistency-based
  selection (Gao et al., 2019, 5 augmentations), BADGE (Ash et al., 2020).
* "Results averaged over six random seeds" (Figure 10 caption).

This script is a thin driver: every piece of selection/training math it
needs already exists in this repo --
``batch_active_learning.mixmatch.MixMatchTrainer`` (MixMatch, matches the
paper's hyperparameters by name), ``batch_active_learning.acquisition``
(all six strategies of Figure 10, including ``bico`` = Eq. (10) with a CNTK
proxy), and ``batch_active_learning.active_learning.ActiveLearningLoop``
(the retrain-from-scratch acquisition loop). This script only wires them to
the paper's exact audio data/preprocessing/architecture/hyperparameters.

Deviations from the paper, called out explicitly:
* The "presample 100 augmentations per labeled point, concatenated for
  batch gradient descent" refinement of the ``bico`` inner problem is NOT
  implemented -- ``batch_active_learning.acquisition.bico`` fits the inner
  logistic regression directly on the (unaugmented) labeled + pseudo-labeled
  proxy features. This is a real simplification versus the paper (the
  augmentation replicas act as a form of data-dependent regularization for
  the inner problem); everything else in the ``bico`` acquisition (CNTK
  proxy, 2048-dim Nystrom features, 1e-4 L2 penalty, 100 CG steps, one-by-
  one forward selection) matches the paper.
* Spoken Digit (Jackson, 2016) has no official PyTorch/torchaudio loader;
  this script clones the "free-spoken-digit-dataset" GitHub repository
  (the dataset's canonical distribution) on first use.
* Speech Commands V2 is loaded via ``torchaudio.datasets.SPEECHCOMMANDS``
  (official V2 release); background-noise segments for augmentation (iv)
  are taken from its ``_background_noise_`` folder, as the paper specifies
  ("we use the noise segments from the Speech Commands data set") even when
  running on Spoken Digit.
* Requires ``torchaudio`` (not otherwise a dependency of this repo) for
  resampling and mel-spectrogram extraction: ``pip install torchaudio``.

COMPUTE COST -- likely the single most expensive reproduction here. Per
seed, EVERY acquisition round (paper: up to ~19 rounds to reach 200 labels
from a start of 10/50, batch 10) retrains a WRN-28-10 from scratch with
MixMatch "until convergence" (Table 7's recipe: 30 epochs, but over a
growing *unlabeled* pool of up to 85000 utterances for Speech Commands, so
each epoch is a full pass over that unlabeled pool too). The ``bico``
acquisition additionally needs a CNTK-6+GAP Nystrom feature map with
q=2048 landmarks recomputed against the current pool at every round -- the
same expensive kernel already used (and OOM-guarded) elsewhere in this
repo. Multiply by 5 acquisition strategies and 6 seeds. Expect a full run
to take days on a single Kaggle GPU; use ``--smoke-test`` first.

Run with::

    python experiments/run_batch_active_learning_paper.py --dataset spoken_digit --smoke-test
    python experiments/run_batch_active_learning_paper.py --dataset spoken_digit
    python experiments/run_batch_active_learning_paper.py --dataset speech_commands
"""

import argparse
import csv
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from batch_active_learning.active_learning import ActiveLearningLoop
from batch_active_learning.mixmatch import MixMatchTrainer


SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
CLIP_LEN = int(SAMPLE_RATE * CLIP_SECONDS)
WIN_MS, HOP_MS, N_MELS = 128, 32, 32


# ----------------------------------------------------------------------
# audio -> fixed-length waveform -> 32x32 mel spectrogram
# ----------------------------------------------------------------------
def fix_length(wav, target_len=CLIP_LEN):
    """Center-pad or center-crop a 1-D waveform to exactly ``target_len`` samples."""
    n = wav.shape[-1]
    if n == target_len:
        return wav
    if n < target_len:
        pad = target_len - n
        left = pad // 2
        return F.pad(wav, (left, pad - left))
    start = (n - target_len) // 2
    return wav[..., start:start + target_len]


def load_spoken_digit(data_root='data', seed=0):
    """Jackson (2016) Free Spoken Digit Dataset: 2700 utterances, 10 classes."""
    import torchaudio

    repo_dir = os.path.join(data_root, 'free-spoken-digit-dataset')
    if not os.path.isdir(repo_dir):
        os.makedirs(data_root, exist_ok=True)
        print('cloning free-spoken-digit-dataset into {}...'.format(repo_dir))
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/Jakobovski/free-spoken-digit-dataset.git', repo_dir],
                       check=True)
    rec_dir = os.path.join(repo_dir, 'recordings')
    files = sorted(f for f in os.listdir(rec_dir) if f.endswith('.wav'))
    waves, labels = [], []
    for fname in files:
        label = int(fname.split('_')[0])
        wav, sr = torchaudio.load(os.path.join(rec_dir, fname))
        wav = wav.mean(dim=0)  # mono
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        waves.append(fix_length(wav))
        labels.append(label)
    X = torch.stack(waves)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y, 10


def load_speech_commands(data_root='data', seed=0, max_per_class=None):
    """Warden (2018) Speech Commands V2: ~85000 utterances, 35 classes."""
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS

    ds = SPEECHCOMMANDS(data_root, download=True)
    # torchaudio's SPEECHCOMMANDS exposes label via __getitem__; build the
    # label vocabulary from the dataset's own label list if available,
    # otherwise infer it from a first pass.
    if hasattr(ds, 'get_metadata'):
        all_labels = sorted({ds.get_metadata(i)[2] for i in range(len(ds))})
    else:
        all_labels = sorted({ds[i][2] for i in range(len(ds))})
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    rs = np.random.RandomState(seed)
    by_class = {l: [] for l in all_labels}
    for i in range(len(ds)):
        wav, sr, label, *_ = ds[i]
        by_class[label].append(i)

    waves, labels = [], []
    noise_waves = []
    for label, inds in by_class.items():
        if label == '_background_noise_':
            for i in inds:
                wav, sr, *_ = ds[i]
                wav = wav.mean(dim=0)
                if sr != SAMPLE_RATE:
                    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
                noise_waves.append(wav)
            continue
        chosen = inds if max_per_class is None else rs.choice(
            inds, min(max_per_class, len(inds)), replace=False)
        for i in chosen:
            wav, sr, *_ = ds[i]
            wav = wav.mean(dim=0)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            waves.append(fix_length(wav))
            labels.append(label_to_idx[label])

    X = torch.stack(waves)
    y = torch.tensor(labels, dtype=torch.long)
    n_classes = len([l for l in all_labels if l != '_background_noise_'])
    noise = torch.stack(noise_waves) if noise_waves else torch.zeros(1, CLIP_LEN)
    return X, y, n_classes, noise


class MelSpectrogramFrontend(nn.Module):
    """Fixed (non-trainable) waveform -> 32x32 log-mel-spectrogram frontend,
    matching "window length of 128 ms, hop length of 32 ms and 32 bins"."""

    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS, win_ms=WIN_MS, hop_ms=HOP_MS):
        super().__init__()
        import torchaudio
        win_length = int(sample_rate * win_ms / 1000)
        hop_length = int(sample_rate * hop_ms / 1000)
        n_fft = 1
        while n_fft < win_length:
            n_fft *= 2
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, wav):
        """``wav``: (B, T) -> (B, 1, n_mels, ~32)."""
        spec = self.to_db(self.mel(wav))  # (B, n_mels, frames)
        spec = F.interpolate(spec.unsqueeze(1), size=(N_MELS, N_MELS), mode='bilinear',
                             align_corners=False)
        mu = spec.mean(dim=(1, 2, 3), keepdim=True)
        sigma = spec.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return (spec - mu) / sigma


class MelWideResNet(nn.Module):
    """WRN-28-10 (wd handled by the optimizer, no dropout) over 32x32 mel
    spectrograms -- Sec. 5.5's architecture, operating end-to-end on raw
    waveforms so that the paper's audio-domain augmentations (applied to
    the waveform) compose naturally with MixMatch's ``augment_fn`` contract.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.frontend = MelSpectrogramFrontend()
        self.backbone = models.WideResNet28_10(num_classes=num_classes, dropout_rate=0.0, in_channels=1)

    def forward(self, wav):
        return self.backbone(self.frontend(wav))

    def embed(self, wav):
        return self.backbone.embed(self.frontend(wav))


# ----------------------------------------------------------------------
# audio augmentations (paper, Sec. 5.5): each applied independently with
# probability 0.5, in order: amplitude, speed, time-shift, background noise
# ----------------------------------------------------------------------
def make_audio_augment_fn(noise_bank, p=0.5):
    def augment(wav):
        b, t = wav.shape
        out = wav.clone()

        mask = torch.rand(b, device=wav.device) < p
        amp = torch.empty(b, device=wav.device).uniform_(0.8, 1.2)
        out[mask] = out[mask] * amp[mask].unsqueeze(-1)

        mask = torch.rand(b, device=wav.device) < p
        if mask.any():
            speeds = np.random.uniform(0.8, 1.2, size=int(mask.sum().item()))
            idx = torch.nonzero(mask, as_tuple=True)[0]
            for j, s in zip(idx.tolist(), speeds):
                n = int(round(t / s))
                resampled = F.interpolate(out[j].view(1, 1, -1), size=n, mode='linear',
                                          align_corners=False).view(-1)
                out[j] = fix_length(resampled, t)

        mask = torch.rand(b, device=wav.device) < p
        if mask.any():
            shifts_ms = np.random.uniform(-250, 250, size=b)
            for j in range(b):
                if not mask[j]:
                    continue
                shift = int(round(shifts_ms[j] * SAMPLE_RATE / 1000))
                out[j] = torch.roll(out[j], shifts=shift)

        mask = torch.rand(b, device=wav.device) < p
        if mask.any() and noise_bank.shape[0] > 0:
            snrs_db = np.random.uniform(0, 40, size=b)
            noise_idx = np.random.randint(0, noise_bank.shape[0], size=b)
            for j in range(b):
                if not mask[j]:
                    continue
                noise = noise_bank[noise_idx[j]].to(wav.device)
                start = np.random.randint(0, max(1, noise.shape[0] - t))
                seg = fix_length(noise[start:start + t], t)
                sig_power = out[j].pow(2).mean().clamp_min(1e-10)
                noise_power = seg.pow(2).mean().clamp_min(1e-10)
                snr_lin = 10.0 ** (snrs_db[j] / 10.0)
                scale = torch.sqrt(sig_power / (snr_lin * noise_power))
                out[j] = out[j] + scale * seg
        return out

    return augment


# ----------------------------------------------------------------------
# CNTK-Nystrom proxy features for the 'bico' acquisition (Eq. (10), q=2048)
# ----------------------------------------------------------------------
def make_cntk_nystrom_feature_fn(X_pool, frontend, q=2048, kernel_batch_size=4, seed=0, device='cpu'):
    import jax_patch  # noqa: F401
    from bicoreset.cntk import build_cntk6_gap_kernel_fn
    from batch_active_learning.proxy import NystromFeatureMap

    with torch.no_grad():
        specs = []
        for start in range(0, X_pool.shape[0], 256):
            xb = X_pool[start:start + 256].to(device)
            specs.append(frontend(xb).cpu().numpy())
        specs = np.concatenate(specs).transpose(0, 2, 3, 1)  # (n, 32, 32, 1)

    cntk = build_cntk6_gap_kernel_fn(batch_size=kernel_batch_size)

    def kernel_fn(x, y):
        return cntk(np.asarray(x).reshape(-1, N_MELS, N_MELS, 1), np.asarray(y).reshape(-1, N_MELS, N_MELS, 1))

    rs = np.random.RandomState(seed)
    q = min(q, specs.shape[0])
    landmark_inds = rs.choice(specs.shape[0], q, replace=False)
    basis = specs[landmark_inds]
    nystrom = NystromFeatureMap(kernel_fn, basis)

    def feature_fn(wav_np):
        wav_t = torch.from_numpy(np.asarray(wav_np)).float()
        with torch.no_grad():
            spec = []
            for start in range(0, wav_t.shape[0], 256):
                spec.append(frontend(wav_t[start:start + 256].to(device)).cpu().numpy())
            spec = np.concatenate(spec).transpose(0, 2, 3, 1)
        return nystrom(spec)

    return feature_fn


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Sec. 5.5 batch active learning reproduction (Table 7, Figure 10)')
    parser.add_argument('--dataset', choices=['spoken_digit', 'speech_commands'], default='spoken_digit')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--methods', default='uniform,max_entropy,kcenter,badge,bico',
                        help='comma-separated subset of uniform,max_entropy,kcenter,consistency,badge,bico '
                             '(consistency is off by default -- "cold start failure" noted by the paper itself)')
    parser.add_argument('--seeds', type=int, default=6, help='paper: six random seeds')
    parser.add_argument('--batch-size', type=int, default=10, help='paper: m = 10')
    parser.add_argument('--budget', type=int, default=200, help='paper: n_labeled <= 200')
    parser.add_argument('--start-spoken-digit', type=int, default=10, help='paper: 10 initial labels')
    parser.add_argument('--start-speech-commands', type=int, default=50, help='paper: 50 initial labels')

    parser.add_argument('--mixmatch-epochs', type=int, default=30, help='Table 7: 30 epochs')
    parser.add_argument('--mixmatch-lr', type=float, default=1e-3, help='Table 7: Adam, lr=1e-3')
    parser.add_argument('--n-augmentations', type=int, default=2, help='paper: two augmentations for label guessing')
    parser.add_argument('--lambda-u', type=float, default=10.0, help='paper: lambda_u = 10')

    parser.add_argument('--nystrom-dim', type=int, default=2048, help='paper: q = 2048')
    parser.add_argument('--bico-inner-reg', type=float, default=1e-4, help='paper: 1e-4 L2 penalty')
    parser.add_argument('--bico-cg-iters', type=int, default=100, help='paper: 100 CG steps')
    parser.add_argument('--kernel-batch-size', type=int, default=4,
                        help='CNTK X/Y chunk size (memory-bound, see bicoreset/cntk.py)')

    parser.add_argument('--speech-commands-max-per-class', type=int, default=None,
                        help='subsample each Speech Commands class to at most this many clips '
                             '(paper uses the full ~85000-utterance set; this is a smoke-test/'
                             'compute-budget knob)')

    parser.add_argument('--smoke-test', action='store_true')

    args = parser.parse_args()
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    if args.smoke_test:
        args.seeds = 1
        args.budget = min(args.budget, 40)
        args.mixmatch_epochs = min(args.mixmatch_epochs, 2)
        args.nystrom_dim = min(args.nystrom_dim, 32)
        args.bico_cg_iters = min(args.bico_cg_iters, 10)
        args.speech_commands_max_per_class = args.speech_commands_max_per_class or 20
        print('*** --smoke-test: seeds={}, budget={}, epochs={}, q={} -- pipeline check only ***'.format(
            args.seeds, args.budget, args.mixmatch_epochs, args.nystrom_dim))

    print('loading {}...'.format(args.dataset))
    if args.dataset == 'spoken_digit':
        X, y, n_classes = load_spoken_digit(args.data_root)
        start_size = args.start_spoken_digit
        # noise for Spoken Digit is still drawn from Speech Commands, per the paper
        # ("we use the noise segments from the Speech Commands data set")
        _, _, _, noise_bank = load_speech_commands(args.data_root, max_per_class=1)
    else:
        X, y, n_classes, noise_bank = load_speech_commands(
            args.data_root, max_per_class=args.speech_commands_max_per_class)
        start_size = args.start_speech_commands

    print('{}: {} utterances, {} classes'.format(args.dataset, X.shape[0], n_classes))

    X_pool, y_pool, X_test, y_test = _train_test_split(X, y, test_frac=0.2, seed=0)
    frontend = MelSpectrogramFrontend().to(args.device)
    augment_fn = make_audio_augment_fn(noise_bank)

    def model_fn():
        return MelWideResNet(n_classes)

    trainer = MixMatchTrainer(
        num_classes=n_classes, augment_fn=augment_fn, n_augmentations=args.n_augmentations,
        temperature=0.5, alpha=0.75, lambda_u=args.lambda_u,
        epochs=args.mixmatch_epochs, batch_size=64, lr=args.mixmatch_lr,
        optimizer_fn=lambda p: torch.optim.Adam(p, lr=args.mixmatch_lr),
        scheduler_fn=lambda opt, epochs: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs),
        device=args.device, verbose=False)

    rounds = int(np.ceil((args.budget - start_size) / args.batch_size))
    methods = args.methods.split(',')

    print('\n=== Table 7: supervised vs semi-supervised, uniform labels ===')
    for label, use_unlabeled in [('supervised w/o augm.', False), ('MixMatch', True)]:
        rs = np.random.RandomState(0)
        init = _stratified_initial_pool(y_pool, start_size, n_classes, rs)
        unlabeled = np.setdiff1d(np.arange(len(y_pool)), init) if use_unlabeled else np.array([], dtype=int)
        m = model_fn()
        base_trainer = trainer if use_unlabeled else MixMatchTrainer(
            num_classes=n_classes, augment_fn=lambda x: x, epochs=args.mixmatch_epochs,
            batch_size=64, lr=args.mixmatch_lr, device=args.device, verbose=False)
        m = base_trainer.train(m, X_pool[init], y_pool[init], X_pool[unlabeled] if use_unlabeled else None)
        acc = base_trainer.accuracy(m, X_test, y_test)
        print('{:<24} n_labeled={:<4} test_acc={:.4f}'.format(label, start_size, acc))

    print('\n=== Figure 10: batch active learning, {} seeds ===' .format(args.seeds))
    all_rows = []
    feature_fn_cache = {}
    for method in methods:
        for seed in range(args.seeds):
            t0 = time.time()
            rs = np.random.RandomState(seed)
            torch.manual_seed(seed)
            init = _stratified_initial_pool(y_pool, start_size, n_classes, rs)

            kwargs = {}
            if method == 'bico':
                if seed not in feature_fn_cache:
                    feature_fn_cache[seed] = make_cntk_nystrom_feature_fn(
                        X_pool, frontend, q=args.nystrom_dim, kernel_batch_size=args.kernel_batch_size,
                        seed=seed, device=args.device)
                kwargs = dict(feature_fn=feature_fn_cache[seed], num_classes=n_classes,
                             inner_reg=args.bico_inner_reg, cg_iters=args.bico_cg_iters,
                             device=args.device)

            loop = ActiveLearningLoop(model_fn, trainer, acquisition=method, acquisition_kwargs=kwargs,
                                      batch_size=args.batch_size, rounds=rounds, seed=seed, verbose=True)
            history = loop.run(X_pool, y_pool, init, X_test, y_test)
            for h in history:
                all_rows.append(dict(method=method, seed=seed, n_labeled=h['n_labeled'],
                                     test_accuracy=h['test_accuracy']))
            print('[{}] seed={} done in {:.1f}s'.format(method, seed, time.time() - t0))

    csv_path = os.path.join(out_dir, 'batch_active_learning_{}.csv'.format(args.dataset))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'seed', 'n_labeled', 'test_accuracy'])
        writer.writeheader()
        writer.writerows(all_rows)
    print('\nsaved {}'.format(csv_path))

    try:
        import matplotlib.pyplot as plt
        import collections
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in all_rows:
            agg[r['method']][r['n_labeled']].append(r['test_accuracy'])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for method, sizes in agg.items():
            xs = sorted(sizes)
            ys = [np.mean(sizes[x]) for x in xs]
            ax.plot(xs, ys, marker='o', label=method)
        ax.set_xlabel('Nr. Labeled Samples')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Batch active learning, {}'.format(args.dataset))
        ax.legend()
        fig.tight_layout()
        png_path = os.path.join(out_dir, 'batch_active_learning_{}.png'.format(args.dataset))
        fig.savefig(png_path, dpi=150)
        print('saved {}'.format(png_path))
    except ImportError:
        pass


def _train_test_split(X, y, test_frac=0.2, seed=0):
    rs = np.random.RandomState(seed)
    n = X.shape[0]
    perm = rs.permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = perm[:cut], perm[cut:]
    return X[tr], y[tr], X[te], y[te]


def _stratified_initial_pool(y_pool, size, n_classes, rs):
    """At least one labeled sample per class, matching the paper's guarantee."""
    y_np = y_pool.numpy()
    chosen = [rs.choice(np.where(y_np == c)[0], 1)[0] for c in range(n_classes)
             if np.any(y_np == c)]
    chosen = np.array(chosen, dtype=int)
    remaining = size - len(chosen)
    if remaining > 0:
        rest = np.setdiff1d(np.arange(len(y_np)), chosen)
        extra = rs.choice(rest, min(remaining, len(rest)), replace=False)
        chosen = np.concatenate([chosen, extra])
    return chosen[:size]


if __name__ == '__main__':
    main()
