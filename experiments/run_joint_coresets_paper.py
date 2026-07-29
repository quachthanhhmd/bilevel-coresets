"""Literal reproduction of Sec. 4.4 / end of Sec. 5.2.3 (Table 2): joint
coresets for WideResNet-16-4 and VGG16, transferred to MobileNetV2.

Paper setup, quoted verbatim:

* Formulation (Eq. (11), Sec. 4.4): "In practice, if the loss magnitudes are
  of the same order, we can set lambda = 1; an additional heuristic for
  solving the problem with (batch) forward selection is to perform the
  selection step alternatingly for each model."
* Experiment (Sec. 5.2.3): "we generate a joint coreset for WideResNet-16-4
  and VGG16 and evaluate the resulting coreset for transferability on
  MobileNetV2 [...] we use a simple heuristic for approximating the solution
  of Equation (11) with lambda = 1: similarly to the previous experiment, we
  generate the coreset by forward greedy selection in batches of 250 by
  alternating the model in each step (i.e., we select a new batch of points
  for the WideResNet, then for VGG16)."
* "The results in Table 2 show that this simple heuristic improves the
  effectiveness of the joint coreset on VGG16 and the transferability to
  MobileNetV2 at the expense of small performance degradation on
  WideResNet."
* Table 2 reports test accuracy of WideResNet-16-4, VGG16 and MobileNetV2 on
  CIFAR-10 and SVHN, each trained on (a) the single-model WRN coreset
  ("BiCo WRN", i.e. the output of
  ``run_neural_network_coresets_paper.py``) and (b) the jointly-built
  WRN+VGG coreset ("BiCo WRN + VGG"), both of size 23000. Training procedure
  ("the same as for the WideResNet") is Appendix C's SGD+momentum, lr=0.1
  cosine-annealed to 0 over 300*n/m epochs, weight_decay=5e-4, dropout 0.4
  for SVHN only, CIFAR-10-only augmentation -- reused verbatim from
  ``run_neural_network_coresets_paper.py`` (imported, not reimplemented).

This script reuses ``bicoreset.joint.JointBilevelCoreset`` (already
implements exactly the "alternate" heuristic described above -- no new
selection math needed) and the WRN-16-4/VGG16/MobileNetV2 architectures and
Appendix C training recipe from ``run_neural_network_coresets_paper.py``.

Because ``JointBilevelCoreset`` scores one model's implicit gradient per
step in 'alternate' mode, its Neumann-series IHVP settings (T=100 terms,
stochastic Hessian) and retrain-from-scratch behavior are configured
per-model exactly as in the single-model experiment.

COMPUTE COST: same order of magnitude as (in fact somewhat more expensive
than) ``run_neural_network_coresets_paper.py``, since every selection step
now retrains and scores *two* networks (WRN-16-4 and VGG16) instead of one,
and the final comparison also retrains MobileNetV2. Use ``--smoke-test``
first; see that script's docstring for the full cost accounting.

Run with::

    python experiments/run_joint_coresets_paper.py --dataset cifar10 --smoke-test
    python experiments/run_joint_coresets_paper.py --dataset cifar10
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

import models
from bicoreset import losses as bico_losses
from bicoreset.direct import BilevelCoreset as DirectBilevelCoreset
from bicoreset.joint import JointBilevelCoreset

from run_neural_network_coresets_paper import (
    load_dataset, make_train_fn, evaluate,
)


def build_coreset(model_fn, train_fn, loss_fn, neumann_terms, hessian_batch_size, device, verbose=True):
    return DirectBilevelCoreset(
        model_fn=model_fn,
        loss_fn=loss_fn,
        train_fn=train_fn,
        ihvp='neumann',
        ihvp_kwargs={'num_terms': neumann_terms},
        hessian_batch_size=hessian_batch_size,
        max_outer_it=0,
        candidate_pool_size=None,
        retrain_from_scratch=True,
        device=device, verbose=verbose, logging_period=1)


def main():
    parser = argparse.ArgumentParser(description='Sec. 4.4 joint coreset reproduction (Table 2)')
    parser.add_argument('--dataset', choices=['cifar10', 'svhn'], default='cifar10')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output-dir', default=None)

    parser.add_argument('--size', type=int, default=23000, help='paper: coresets of size 23000 (Table 2)')
    parser.add_argument('--selection-batch', type=int, default=250, help='paper: batches of 250')
    parser.add_argument('--start-size', type=int, default=2500,
                        help='not explicitly restated for this experiment; reused from Sec. 5.2.3')
    parser.add_argument('--lam', type=float, default=1.0, help='paper: lambda = 1')
    parser.add_argument('--neumann-terms', type=int, default=100, help='paper: T = 100 (Sec. 5.2.3)')
    parser.add_argument('--hessian-batch-size', type=int, default=128, help='NOT stated by the paper, see run_neural_network_coresets_paper.py')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--base-lr', type=float, default=0.1)
    parser.add_argument('--epoch-scale', type=int, default=300)
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--smoke-test', action='store_true')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    train_pool_size = None
    size = args.size
    if args.smoke_test:
        train_pool_size = 3000
        size = min(size, 400)
        args.start_size = min(args.start_size, 100)
        args.selection_batch = min(args.selection_batch, 100)
        args.neumann_terms = min(args.neumann_terms, 5)
        args.hessian_batch_size = min(args.hessian_batch_size, 64)
        args.epoch_scale = min(args.epoch_scale, 10)
        print('*** --smoke-test: pool={}, size={}, T={} -- pipeline check only ***'.format(
            train_pool_size, size, args.neumann_terms))

    print('loading {}...'.format(args.dataset))
    X_train, y_train, X_test, y_test = load_dataset(args.dataset, args.data_root, train_pool_size, args.seed)
    n_full = X_train.shape[0]
    n_classes = int(y_train.max().item()) + 1
    dropout_rate = 0.4 if args.dataset == 'svhn' else 0.0
    print('train pool: {}, test: {}, classes: {}'.format(n_full, X_test.shape[0], n_classes))

    def wrn_fn():
        return models.WideResNet16_4(num_classes=n_classes, dropout_rate=dropout_rate)

    def vgg_fn():
        return models.VGG16(num_classes=n_classes)

    def mobilenet_fn():
        return models.MobileNetV2(num_classes=n_classes)

    train_fn = make_train_fn(args.dataset, n_full, batch_size=args.train_batch_size,
                             momentum=args.momentum, weight_decay=args.weight_decay,
                             base_lr=args.base_lr, epoch_scale=args.epoch_scale,
                             device=args.device, verbose=True)

    # ------------------------------------------------------------------
    # (a) single-model WRN coreset -- "BiCo WRN" column of Table 2
    # ------------------------------------------------------------------
    print('\n=== building single-model WRN-16-4 coreset (size={}) ===' .format(size))
    t0 = time.time()
    bc_wrn = build_coreset(wrn_fn, train_fn, bico_losses.cross_entropy,
                           args.neumann_terms, args.hessian_batch_size, args.device)
    wrn_inds, _ = bc_wrn.build(X_train, y_train, size, strategy='forward',
                              selection_batch_size=args.selection_batch, start_size=args.start_size)
    print('single-model WRN coreset built in {:.1f}s'.format(time.time() - t0))

    # ------------------------------------------------------------------
    # (b) joint WRN + VGG16 coreset, alternating selection -- "BiCo WRN + VGG"
    # ------------------------------------------------------------------
    print('\n=== building joint WRN-16-4 + VGG16 coreset (size={}, alternating) ===' .format(size))
    t0 = time.time()
    bc_wrn2 = build_coreset(wrn_fn, train_fn, bico_losses.cross_entropy,
                            args.neumann_terms, args.hessian_batch_size, args.device)
    bc_vgg = build_coreset(vgg_fn, train_fn, bico_losses.cross_entropy,
                           args.neumann_terms, args.hessian_batch_size, args.device)
    joint = JointBilevelCoreset([bc_wrn2, bc_vgg], lambdas=[1.0, args.lam], mode='alternate', verbose=True)
    joint_inds, _ = joint.build(X_train, y_train, size, start_size=args.start_size)
    print('joint WRN+VGG coreset built in {:.1f}s'.format(time.time() - t0))

    # ------------------------------------------------------------------
    # evaluate WRN-16-4, VGG16, MobileNetV2 trained on each coreset (Table 2)
    # ------------------------------------------------------------------
    architectures = {'WideResNet-16-4': wrn_fn, 'VGG16': vgg_fn, 'MobileNetV2': mobilenet_fn}
    rows = []
    for coreset_name, inds in [('BiCo WRN', wrn_inds), ('BiCo WRN + VGG', joint_inds)]:
        for arch_name, arch_fn in architectures.items():
            model = arch_fn()
            train_fn(model, X_train[inds], y_train[inds], None)
            acc = evaluate(model, X_test, y_test, args.device)
            print('{:<16} {:<16} test_acc={:.4f}'.format(coreset_name, arch_name, acc))
            rows.append(dict(coreset=coreset_name, architecture=arch_name, test_accuracy=acc))

    csv_path = os.path.join(out_dir, 'joint_coresets_table2_{}.csv'.format(args.dataset))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['coreset', 'architecture', 'test_accuracy'])
        writer.writeheader()
        writer.writerows(rows)
    print('\nsaved {}'.format(csv_path))


if __name__ == '__main__':
    main()
