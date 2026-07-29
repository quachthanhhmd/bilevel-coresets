"""Algorithm 1 directly on the target ConvNet -- on FashionMNIST.

This is the FashionMNIST counterpart of ``demo_direct_coreset.py``: same
algorithm (Sec. 3.3/3.5.1 of the JMLR paper), but on the actual image data set
you already used in ``create_demo_fashion.py`` / ``demo_fashion.ipynb``,
instead of the synthetic blobs used there for a fast CPU sanity check.

The key difference from ``demo_fashion.ipynb``: that notebook builds the
coreset on a Neural Tangent Kernel *proxy* of a CNN (via ``bilevel_coreset.py``
+ jax/neural-tangents, Sec. 3.5.2). Here, ``bicoreset.direct.BilevelCoreset``
differentiates through ``models.ConvNet`` itself -- no proxy, no jax, exactly
the "constructing coresets directly for the target models, without a proxy"
contribution of the journal paper. It is also compared against the same
baselines already available in the repo (uniform, Sensitivity Coreset,
GLISTER, from ``cl_streaming/summary.py``).

Run with::

    python demos/demo_fashion_direct.py

Tunables (edit the constants in ``main()``):
    CORESET_SIZE     -- number of images to keep (50, as in demo_fashion.ipynb)
    CANDIDATE_POOL    -- number of images the selection algorithms get to see
                         (limits cost, same trick as ``limit = 2500`` there)
    CANDIDATE_PER_STEP -- how many of those are scored at each selection step
                         (bicoreset.direct's ``candidate_pool_size``)
    On a CPU this takes a few minutes for CANDIDATE_POOL=1000; on a Kaggle GPU
    it comfortably handles CANDIDATE_POOL=2500 like the notebook.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models
from bicoreset import losses
from bicoreset.direct import BilevelCoreset
from cl_streaming import summary


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_fashion_mnist(data_root='data'):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # FashionMNIST mean/std
    ])
    train = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
    return train, test


def train_convnet(dataset, nr_classes=10, nr_epochs=1000, batch_size=128, device='cpu', lr=5e-4):
    """Same training routine as ``demo_fashion.ipynb``'s ``train_model``."""
    model = models.ConvNet(output_dim=nr_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(nr_epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
    return model


def evaluate(model, dataset, device='cpu', batch_size=512):
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            correct += (model(data).argmax(dim=1) == target).sum().item()
    return correct / len(dataset)


def main():
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    CORESET_SIZE = 50
    CANDIDATE_POOL = 1000          # lower this (e.g. 300) for a quick CPU smoke test
    CANDIDATE_PER_STEP = 300       # candidates scored per selection step
    SELECTION_BATCH = 5

    train_dataset, test_dataset = load_fashion_mnist()
    print('training set: {} images, test set: {} images'.format(len(train_dataset), len(test_dataset)))

    # --- restrict the candidate pool for the selection algorithms, exactly the
    # `limit = 2500` trick of demo_fashion.ipynb: scoring 60k candidates at every
    # step would be far too slow for a demo. The uniform baseline does not need
    # to score anything, so it still samples from the full training set.
    loader = DataLoader(train_dataset, batch_size=CANDIDATE_POOL, shuffle=False)
    X_pool, y_pool = next(iter(loader))

    # ---------------------------------------------------------------
    # 1. uniform baseline
    # ---------------------------------------------------------------
    uniform_inds = np.random.RandomState(1).choice(len(train_dataset), CORESET_SIZE, replace=False)
    t0 = time.time()
    net_uniform = train_convnet(Subset(train_dataset, uniform_inds), device=device)
    acc_uniform = evaluate(net_uniform, test_dataset, device=device)
    print('uniform            (m={}): {:.4f}  [{:.1f}s]'.format(CORESET_SIZE, acc_uniform, time.time() - t0))

    # ---------------------------------------------------------------
    # 2. Sensitivity Coreset baseline (cl_streaming/summary.py)
    # ---------------------------------------------------------------
    rs = np.random.RandomState(0)
    sens_inds = summary.Summarizer.factory('sensitivity', rs).build_summary(
        X_pool.numpy(), y_pool.numpy(), CORESET_SIZE)
    t0 = time.time()
    net_sens = train_convnet(Subset(train_dataset, sens_inds), device=device)
    acc_sens = evaluate(net_sens, test_dataset, device=device)
    print('sensitivity coreset (m={}): {:.4f}  [{:.1f}s]'.format(CORESET_SIZE, acc_sens, time.time() - t0))

    # ---------------------------------------------------------------
    # 3. GLISTER baseline (cl_streaming/summary.py)
    # ---------------------------------------------------------------
    net_for_glister = models.ConvNet(output_dim=10).to(device)
    glister_inds = summary.Summarizer.factory('glister', rs).build_summary(
        X_pool.numpy(), y_pool.numpy(), CORESET_SIZE, model=net_for_glister, device=device)
    t0 = time.time()
    net_glister = train_convnet(Subset(train_dataset, glister_inds), device=device)
    acc_glister = evaluate(net_glister, test_dataset, device=device)
    print('GLISTER             (m={}): {:.4f}  [{:.1f}s]'.format(CORESET_SIZE, acc_glister, time.time() - t0))

    # ---------------------------------------------------------------
    # 4. BiCo directly on the target ConvNet -- no proxy, no jax (Sec. 3.3/3.5.1)
    # ---------------------------------------------------------------
    builder = BilevelCoreset(
        model_fn=lambda: models.ConvNet(output_dim=10),
        loss_fn=losses.cross_entropy,
        inner_reg=1e-4,
        ihvp='neumann', ihvp_kwargs={'num_terms': 50, 'alpha': 0.01, 'damping': 1e-3},
        max_inner_it=300, inner_lr=5e-4,
        max_outer_it=0,                    # binary (unweighted) coreset
        candidate_pool_size=CANDIDATE_PER_STEP,
        outer_batch_size=256,
        hessian_batch_size=64,              # stochastic Hessian, Sec. 5.2.3
        retrain_from_scratch=True,
        device=device, verbose=True, logging_period=1)

    print('\nbuilding the direct BiCo coreset ({} candidates, {} selected)...'.format(
        CANDIDATE_POOL, CORESET_SIZE))
    t0 = time.time()
    inds, weights = builder.build(
        X_pool, y_pool, CORESET_SIZE,
        strategy='forward', selection_batch_size=SELECTION_BATCH, start_size=SELECTION_BATCH)
    build_time = time.time() - t0

    net_coreset = train_convnet(Subset(train_dataset, inds), device=device)
    acc_coreset = evaluate(net_coreset, test_dataset, device=device)
    print('BiCo direct         (m={}): {:.4f}  [selection {:.1f}s]'.format(
        CORESET_SIZE, acc_coreset, build_time))

    # ---------------------------------------------------------------
    # summary
    # ---------------------------------------------------------------
    print('\n{:<20}{:>10}{:>28}'.format('method', 'accuracy', 'improvement vs uniform'))
    results = {'uniform': acc_uniform, 'sensitivity': acc_sens, 'glister': acc_glister,
              'BiCo direct': acc_coreset}
    for name, acc in results.items():
        print('{:<20}{:>10.2%}{:>28.2%}'.format(name, acc, acc - acc_uniform))

    print('\nclass distribution (per-class counts, 10 classes):')
    print('uniform    ', np.bincount(np.asarray(train_dataset.targets)[uniform_inds], minlength=10))
    print('sensitivity', np.bincount(y_pool.numpy()[sens_inds], minlength=10))
    print('glister    ', np.bincount(y_pool.numpy()[glister_inds], minlength=10))
    print('BiCo direct', np.bincount(y_pool.numpy()[inds], minlength=10))


if __name__ == '__main__':
    main()
