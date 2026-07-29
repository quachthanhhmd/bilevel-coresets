import numpy as np
import json
import argparse
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_local_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def parse_csv_list(value, cast=None):
    if value is None:
        return None
    items = [item.strip() for item in value.split(',') if item.strip()]
    if cast is not None:
        return [cast(item) for item in items]
    return items


def get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=False, path='cl_results',
                   save_path='cl_results/best_betas.txt'):
    path = resolve_local_path(path)
    save_path = resolve_local_path(save_path)
    best_betas = {}
    for method in methods:
        best_beta_for_method = {}
        for dataset in datasets:
            best_acc, best_beta = -1, -1
            for beta in betas:
                res = []
                for seed in seeds:
                    file_path = '{}/{}_{}_{}_{}_{}.txt'.format(path, dataset, method, buffer_size, beta, seed)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            res.append(data['test_acc'])
                    except FileNotFoundError:
                        continue
                if len(res) > 0 and np.mean(res) > best_acc:
                    best_acc = np.mean(res)
                    best_beta = beta
            if best_beta == -1:
                print('WARNING: no result files found for {} {} in {}'.format(method, dataset, path))
            print(method, dataset, best_beta)
            best_beta_for_method[dataset] = best_beta
        best_betas[method] = best_beta_for_method
    if save_best:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(best_betas, f, sort_keys=True, indent=4)
    return best_betas


def get_result(method, dataset, beta, seeds, buffer_size, path='cl_results'):
    path = resolve_local_path(path)
    res = []
    for seed in seeds:
        file_path = '{}/{}_{}_{}_{}_{}.txt'.format(path, dataset, method, buffer_size, beta, seed)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                res.append(data)
        except FileNotFoundError:
            continue
    return res


def continual_learning_results(datasets=None, methods=None, seeds=None, betas=None, buffer_size=100):
    if datasets is None:
        datasets = ['permmnist', 'splitmnist']
    if methods is None:
        methods = [
        'uniform', 'kmeans_features', 'kmeans_embedding', 'kmeans_grads',
        'kcenter_features', 'kcenter_embedding', 'kcenter_grads',
        'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching',
        'coreset'
    ]
    if seeds is None:
        seeds = list(range(5))
    if betas is None:
        betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    best_betas = get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=True, path='cl_results',
                                save_path='cl_results/best_betas.txt')
    print('Continual Learning study\n')

    print('Method \\ Dataset'.ljust(45), end='')
    for dataset in datasets:
        print(' ' + dataset.ljust(18), end='')
    print('')
    for method in methods:
        print(method.ljust(43), end='')
        for dataset in datasets:
            beta = best_betas[method][dataset]
            res = get_result(method, dataset, beta, seeds, buffer_size, 'cl_results')
            res = [r['test_acc'] for r in res]
            if len(res) == 0:
                print(' N/A'.ljust(20), end='')
            else:
                print(' {:.2f} +- {:.2f}'.format(np.mean(res), np.std(res)).ljust(20), end='')
        print('')


def streaming_results():
    datasets = ['permmnist', 'splitmnist']
    methods = ['reservoir', 'coreset']
    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 100

    best_betas = get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=True, path='streaming_results',
                                save_path='streaming_results/best_betas.txt')
    print('Streaming study\n')

    print('Method \\ Dataset'.ljust(45), end='')
    for dataset in datasets:
        print(' ' + dataset.ljust(18), end='')
    print('')
    for method in methods:
        print(method.ljust(43), end='')
        for dataset in datasets:
            beta = best_betas[method][dataset]
            res = get_result(method, dataset, beta, seeds, buffer_size, 'streaming_results')
            res = [r['test_acc'] for r in res]
            print(' {:.2f} +- {:.2f}'.format(np.mean(res), np.std(res)).ljust(20), end='')
        print('')


def imbalanced_streaming_results():
    datasets = ['splitmnistimbalanced']
    methods = ['reservoir', 'cbrs', 'coreset']
    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 100

    best_betas = get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=True, path='streaming_results',
                                save_path='streaming_results/best_betas_imbalanced.txt')
    print('Streaming study\n')

    print('Method \\ Dataset'.ljust(45), end='')
    for dataset in datasets:
        print(' ' + dataset.ljust(18), end='')
    print('')
    for method in methods:
        print(method.ljust(43), end='')
        for dataset in datasets:
            beta = best_betas[method][dataset]
            res = get_result(method, dataset, beta, seeds, buffer_size, 'streaming_results')
            res = [r['test_acc'] for r in res]
            print(' {:.2f} +- {:.2f}'.format(np.mean(res), np.std(res)).ljust(20), end='')
        print('')


def splitcifar_results():
    datasets = ['splitcifar']
    methods = [
        'uniform', 'kmeans_features', 'kmeans_embedding', 'kmeans_grads',
        'kcenter_features', 'kcenter_embedding', 'kcenter_grads',
        'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching',
        'coreset'
    ]

    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 200

    best_betas = get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=True, path='cl_results',
                                save_path='cl_results/best_betas_splitcifar.txt')
    print('Streaming study\n')

    print('Method \\ Dataset'.ljust(45), end='')
    for dataset in datasets:
        print(' ' + dataset.ljust(18), end='')
    print('')
    for method in methods:
        print(method.ljust(43), end='')
        for dataset in datasets:
            beta = best_betas[method][dataset]
            res = get_result(method, dataset, beta, seeds, buffer_size, 'cl_results')
            res = [r['test_acc'] for r in res]
            print(' {:.2f} +- {:.2f}'.format(np.mean(res), np.std(res)).ljust(20), end='')
        print('')


def imbalanced_streaming_cifar_results():
    datasets = ['stream_imbalanced_splitcifar']
    methods = ['reservoir', 'cbrs', 'coreset']
    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 200

    best_betas = get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=True, path='streaming_results',
                                save_path='streaming_results/best_betas_imbalanced_cifar.txt')
    print('Streaming study\n')

    print('Method \\ Dataset'.ljust(45), end='')
    for dataset in datasets:
        print(' ' + dataset.ljust(18), end='')
    print('')
    for method in methods:
        print(method.ljust(43), end='')
        for dataset in datasets:
            beta = best_betas[method][dataset]
            res = get_result(method, dataset, beta, seeds, buffer_size, 'streaming_results')
            res = [r['test_acc'] for r in res]
            print(' {:.2f} +- {:.2f}'.format(np.mean(res), np.std(res)).ljust(20), end='')
        print('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Results processor')
    parser.add_argument('--exp', default='cl',
                        choices=['cl', 'streaming', 'imbalanced_streaming', 'splitcifar', 'imbalanced_streaming_cifar'])
    parser.add_argument('--datasets', default=None,
                        help='Comma-separated datasets override (e.g. permmnist,splitmnist)')
    parser.add_argument('--methods', default=None,
                        help='Comma-separated methods override (e.g. uniform,coreset)')
    parser.add_argument('--seeds', default=None,
                        help='Comma-separated seeds override (e.g. 0,1,2)')
    parser.add_argument('--betas', default=None,
                        help='Comma-separated betas override (e.g. 0.01,1.0,100.0)')
    parser.add_argument('--buffer_size', default=None, type=int,
                        help='Override buffer size used to locate result files')
    args = parser.parse_args()
    exp = args.exp
    if exp == 'cl':
        continual_learning_results(
            datasets=parse_csv_list(args.datasets),
            methods=parse_csv_list(args.methods),
            seeds=parse_csv_list(args.seeds, cast=int),
            betas=parse_csv_list(args.betas, cast=float),
            buffer_size=args.buffer_size if args.buffer_size is not None else 100
        )
    elif exp == 'streaming':
        streaming_results()
    elif exp == 'imbalanced_streaming':
        imbalanced_streaming_results()
    elif exp == 'splitcifar':
        splitcifar_results()
    elif exp == 'imbalanced_streaming_cifar':
        imbalanced_streaming_cifar_results()
    else:
        raise Exception('Unknown experiment')
