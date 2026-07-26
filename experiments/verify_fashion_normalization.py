"""Verify the FashionMNIST normalization constants used in datagen.py.

``cl_streaming/datagen.py``'s ``fashion_mnist_transforms()`` normalizes every
image with ``transforms.Normalize((0.2860,), (0.3530,))`` -- these are the
commonly-cited per-pixel mean/std of the FashionMNIST *training* set, computed
over the raw ``[0, 1]`` pixel values produced by ``ToTensor()``, before
normalization is applied. This script recomputes those two numbers directly
from the real training set (all 60000 images, exact -- not a sample estimate)
and compares them to the constants, producing a report-ready figure.

Needs torchvision + internet (e.g. on Kaggle); this sandbox has neither, so
this could only be validated here against synthetic data of known mean/std,
not run against the real FashionMNIST files.

Run with::

    python experiments/verify_fashion_normalization.py
"""

import argparse
import os

import numpy as np
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# The constants used by cl_streaming/datagen.py's fashion_mnist_transforms().
CODE_MEAN = 0.2860
CODE_STD = 0.3530


def load_fashion_mnist_train(data_root='data'):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    return datasets.FashionMNIST(root=data_root, train=True, download=True,
                                 transform=transforms.ToTensor())


def compute_stats_and_sample(train_dataset, batch_size=1000, sample_batches_for_hist=2, seed=0):
    """Exact mean/std over every pixel of the training set (single pass), plus
    a subsample of raw pixel values (from the first ``sample_batches_for_hist``
    shuffled batches) for the histogram panel."""
    import torch
    torch.manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    n_pixels = 0
    total = 0.0
    total_sq = 0.0
    hist_pixels = []
    for i, (images, _) in enumerate(loader):
        x = images.view(images.size(0), -1).double()
        n_pixels += x.numel()
        total += x.sum().item()
        total_sq += (x ** 2).sum().item()
        if i < sample_batches_for_hist:
            hist_pixels.append(images.view(-1).numpy())

    mean = total / n_pixels
    var = total_sq / n_pixels - mean ** 2
    std = var ** 0.5
    pixels = np.concatenate(hist_pixels)
    return mean, std, pixels, n_pixels


def plot(mean, std, pixels, n_images, output_path):
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel 1: pixel-value histogram with the computed mean/std overlaid.
    ax = axes[0]
    ax.hist(pixels, bins=100, color='#2a78d6', alpha=0.85, density=True)
    ax.axvline(mean, color='#e34948', linestyle='--', linewidth=2,
              label='Mean tính được = {:.4f}'.format(mean))
    ax.axvline(mean - std, color='#e58a1a', linestyle=':', linewidth=1.5,
              label='Mean ± Std = {:.4f} ± {:.4f}'.format(mean, std))
    ax.axvline(mean + std, color='#e58a1a', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Giá trị pixel (sau ToTensor(), thang [0, 1])')
    ax.set_ylabel('Mật độ')
    ax.set_title('Phân bố giá trị pixel FashionMNIST (train)')
    ax.legend(fontsize=9)

    # Panel 2: computed (from the real data) vs. the constants used in code.
    ax = axes[1]
    labels = ['Mean', 'Std']
    computed = [mean, std]
    code_vals = [CODE_MEAN, CODE_STD]
    x = np.arange(2)
    w = 0.32
    bars1 = ax.bar(x - w / 2, computed, width=w, color='#008300',
                   edgecolor='black', label='Tính từ {} ảnh train thật'.format(n_images))
    bars2 = ax.bar(x + w / 2, code_vals, width=w, color='#4a3aa7',
                   edgecolor='black', label='Hằng số trong datagen.py')
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    '{:.4f}'.format(bar.get_height()), ha='center', va='bottom',
                    fontsize=10, fontweight='semibold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(computed + code_vals) * 1.3)
    ax.set_title('So sánh: tính từ dữ liệu thật vs. hằng số trong code')
    ax.legend(fontsize=9)

    fig.suptitle('Kiểm chứng hằng số chuẩn hóa FashionMNIST '
                 '(transforms.Normalize((0.2860,), (0.3530,)))', y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print('Saved plot to {}'.format(output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'fashion_normalization_verification.png')

    train = load_fashion_mnist_train(args.data_root)
    mean, std, pixels, n_pixels = compute_stats_and_sample(train)
    n_images = len(train)

    diff_mean = abs(mean - CODE_MEAN)
    diff_std = abs(std - CODE_STD)
    print('Tính từ toàn bộ {} ảnh train FashionMNIST ({} pixel, thang [0, 1]):'.format(n_images, n_pixels))
    print('  mean = {:.6f}   (code dùng {}, lệch {:.6f} = {:.3f}%)'.format(
        mean, CODE_MEAN, diff_mean, 100 * diff_mean / CODE_MEAN))
    print('  std  = {:.6f}   (code dùng {}, lệch {:.6f} = {:.3f}%)'.format(
        std, CODE_STD, diff_std, 100 * diff_std / CODE_STD))

    plot(mean, std, pixels, n_images, output_path)


if __name__ == '__main__':
    main()
