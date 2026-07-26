import argparse
import torch
import numpy as np
import random as rnd
import os
import json
import time
from torch.utils.data import DataLoader
import loss_utils
import models
import bilevel_coreset
from bicoreset.direct import BilevelCoreset as DirectBilevelCoreset
from bicoreset import losses as bico_losses
from cl_streaming import summary
from cl_streaming import datagen
from cl_streaming import training
# jax_patch and ntk_generator are only needed by method == 'coreset' (the NTK-proxy
# construction); importing them lazily lets 'coreset_direct' and all other methods
# run on a plain torch install, without jax/neural-tangents.

datasets = ['permmnist', 'splitmnist', 'splitfashionmnist']
# 'coreset' builds the coreset on an NTK proxy (bilevel_coreset.py, Sec. 3.5.2 of the
# JMLR paper); 'coreset_direct' builds it directly on the target model being trained
# (bicoreset.direct.BilevelCoreset, Sec. 3.3/3.5.1) -- no proxy, but much slower since
# it retrains the model at every forward-selection step.
methods = ['uniform', 'coreset', 'coreset_direct',
           'kmeans_features', 'kcenter_features', 'kmeans_grads',
           'kmeans_embedding', 'kcenter_embedding', 'kcenter_grads',
           'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching',
           'sensitivity', 'glister']


def get_kernel_fn(dataset):
    import jax_patch  # noqa: F401  (hot-patches jax; must be imported before neural_tangents)
    from cl_streaming import ntk_generator
    if dataset == 'permmnist':
        return lambda x, y: ntk_generator.generate_fnn_ntk(x.reshape(-1, 28 * 28), y.reshape(-1, 28 * 28))
    else:
        return lambda x, y: ntk_generator.generate_cnn_ntk(x.reshape(-1, 28, 28, 1), y.reshape(-1, 28, 28, 1))


def continual_learning(args):
    nr_epochs = args.nr_epochs
    beta = args.beta
    dataset = args.dataset
    device = args.device
    method = args.method
    samples_per_task = args.samples_per_task
    buffer_size = args.buffer_size
    num_workers = args.num_workers
    pin_memory = device == 'cuda'
    if dataset == 'permmnist':
        generator = datagen.PermutedMnistGenerator(samples_per_task)
    elif dataset == 'splitmnist':
        generator = datagen.SplitMnistGenerator(samples_per_task)
    elif dataset == 'splitfashionmnist':
        generator = datagen.SplitFashionMnistGenerator(samples_per_task)

    tasks = []
    train_loaders = []
    test_loaders = []
    for i in range(generator.max_iter):
        X_train, y_train, X_test, y_test = generator.next_task()
        tasks.append((X_train, y_train, X_test, y_test))
        train_data = datagen.NumpyDataset(X_train, y_train)
        train_loaders.append(
            DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                       pin_memory=pin_memory))
        test_data = datagen.NumpyDataset(X_test, y_test)
        test_loaders.append(
            DataLoader(test_data, batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory))

    nr_classes = 10
    inner_reg = 1e-3

    if dataset == 'permmnist':
        model = models.FNNet(28 * 28, 100, nr_classes).to(device)
    else:
        model = models.ConvNet(nr_classes).to(device)

    training_op = training.Training(model, device, nr_epochs, beta=beta)
    # computed lazily on first use inside the loop -- only method == 'coreset' needs it
    kernel_fn = get_kernel_fn(dataset) if method == 'coreset' else None

    bc = bilevel_coreset.BilevelCoreset(outer_loss_fn=loss_utils.cross_entropy,
                                        inner_loss_fn=loss_utils.cross_entropy, out_dim=10, max_outer_it=1,
                                        max_inner_it=200, logging_period=1000)

    # 'coreset_direct': same Algorithm 1 selection rule as 'coreset', but the inner
    # problem is solved on the actual target model (models.FNNet/ConvNet) instead of
    # a kernel proxy. model_fn is re-instantiated at every forward-selection step
    # (retrain_from_scratch=True), which is why this is much slower per task than
    # the proxy-based 'coreset'.
    if dataset == 'permmnist':
        direct_model_fn = lambda: models.FNNet(28 * 28, 100, nr_classes).to(device)
    else:
        direct_model_fn = lambda: models.ConvNet(nr_classes).to(device)
    direct_bc = DirectBilevelCoreset(
        model_fn=direct_model_fn,
        loss_fn=bico_losses.cross_entropy,
        inner_reg=inner_reg,
        ihvp='neumann', ihvp_kwargs={'num_terms': 30, 'alpha': 0.01, 'damping': 1e-3},
        max_inner_it=100, inner_lr=5e-4,
        max_outer_it=0,                 # binary (unweighted) coreset, Sec. 3.5.1
        candidate_pool_size=300,        # candidates scored per selection step
        hessian_batch_size=64,          # stochastic Hessian, Sec. 5.2.3
        retrain_from_scratch=True,
        device=device, verbose=False)

    rs = np.random.RandomState(args.seed)
    
    # Ma trận lưu độ chính xác: acc_matrix[i][j] là độ chính xác trên task j sau khi học xong task i
    acc_matrix = np.zeros((generator.max_iter, generator.max_iter))
    
    start_time = time.time()

    for i in range(generator.max_iter):
        training_op.train(train_loaders[i])
        size_per_task = buffer_size // (i + 1)
        for j in range(i):
            (X, y), w = training_op.buffer[j]
            X, y = X[:size_per_task], y[:size_per_task]
            training_op.buffer[j] = ((X, y), np.ones(len(y)))
        X, y, _, _ = tasks[i]
        if method == 'coreset':
            chosen_inds, _, = bc.build_with_representer_proxy_batch(X, y, size_per_task, kernel_fn, cache_kernel=True,
                                                                    start_size=1, inner_reg=inner_reg)
        elif method == 'coreset_direct':
            selection_batch = max(1, size_per_task // 10)
            chosen_inds, _ = direct_bc.build(
                torch.from_numpy(X).float(), torch.from_numpy(y).long(), size_per_task,
                strategy='forward', selection_batch_size=selection_batch,
                start_size=min(selection_batch, size_per_task))
        else:
            summarizer = summary.Summarizer.factory(method, rs)
            chosen_inds = summarizer.build_summary(X, y, size_per_task, method=method, model=model, device=device)
        X, y = X[chosen_inds], y[chosen_inds]
        assert (X.shape[0] == size_per_task)
        training_op.buffer.append(((X, y), np.ones(len(y))))
        
        # Đánh giá sau mỗi task để vẽ ma trận Forgetting
        for k in range(i + 1):
            acc_matrix[i][k] = training_op.test(test_loaders[k])

    execution_time = time.time() - start_time

    result = []
    for k in range(generator.max_iter):
        result.append(training_op.test(test_loaders[k]))
    filename = '{}_{}_{}_{}_{}.txt'.format(dataset, method, buffer_size, beta, seed)
    if not os.path.exists('cl_results'):
        os.makedirs('cl_results')
    with open('cl_results/' + filename, 'w') as outfile:
        json.dump({
            'test_acc': np.mean(result), 
            'acc_per_task': result,
            'acc_matrix': acc_matrix.tolist(),
            'execution_time': execution_time
        }, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continual Learning')
    parser.add_argument('--seed', type=int, default=0, metavar='seed',
                        help='random seed (default: 0)')
    parser.add_argument('--nr_epochs', default=400, type=int)
    parser.add_argument('--beta', default=1, type=float, help='the buffer penalty')
    parser.add_argument('--dataset', default='splitmnist', choices=datasets)
    parser.add_argument('--method', default='coreset', choices=methods)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--samples_per_task', default=1000, type=int)
    parser.add_argument('--buffer_size', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    args = parser.parse_args()
    print(args)
    seed = args.seed

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)

    continual_learning(args)
