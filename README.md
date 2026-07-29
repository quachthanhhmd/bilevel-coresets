# Coresets via Bilevel Optimization

<img src="thumbnail.png" width="300"/>

This is the reference implementation for "Coresets via Bilevel Optimization for Continual Learning and Streaming" [https://arxiv.org/pdf/2006.03875.pdf](https://arxiv.org/pdf/2006.03875.pdf). 

This repository also contains the implementation of the selection via Nyström proxy used for selecting
batches in "Semi-supervised Batch Active Learning via Bilevel Optimization" [https://arxiv.org/pdf/2010.09654](https://arxiv.org/pdf/2010.09654).
Selection via the Nyström proxy supports data augmentation, it is faster for larger coresets and hence supersedes the
representer proxy in data summarization scenarios.

The [`bicoreset/`](bicoreset) package additionally implements the extensions introduced in the journal version,
"Data Summarization via Bilevel Optimization", JMLR 25 (2024) 1-53 — most importantly coreset construction
**directly on the target model, without a proxy**. See [Journal extensions](#journal-extensions-jmlr-2024) below.

## Overview
To get started with the library, check out [`demo.ipynb`](https://github.com/zalanborsos/bilevel_coresets/blob/main/demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zalanborsos/bilevel_coresets/blob/main/demo.ipynb)
 that shows how to build coresets for a toy regression 
problem and for MNIST classification. The following snippet outlines the general usage:
```python
import bilevel_coreset
import loss_utils
import numpy as np

x, y = load_data()

# define proxy kernel function
linear_kernel_fn = lambda x1, x2: np.dot(x1, x2.T)

coreset_size = 10

coreset_constructor = bilevel_coreset.BilevelCoreset(outer_loss_fn=loss_utils.cross_entropy,
                                                    inner_loss_fn=loss_utils.cross_entropy,
                                                    out_dim=y.shape[1])
coreset_inds, coreset_weights = coreset_constructor.build_with_representer_proxy_batch(x, y, 
                                                    coreset_size, linear_kernel_fn, inner_reg=1e-3)
x_coreset, y_coreset = x[coreset_inds], y[coreset_inds]
```
**Note**: if you are planning to use the library on your problem, the most important hyperparameter to tune
is ```inner_reg```, the regularizer of the inner objective in the representer proxy - 
try the grid [10<sup>-2</sup>, 10<sup>-3</sup>, 10<sup>-4</sup>, 10<sup>-5</sup>, 10<sup>-6</sup>].

## Requirements

Python 3 is required.  To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
If you are planning to use the NTK proxy, consider installing the GPU version of JAX: instructions [here](https://github.com/google/jax#installation).
If you would like to run the experiments, add the project root to your PYTHONPATH env variable.

## Data Summarization

Change dir to ```data_summarization```. For running and plotting the **MNIST summarization** experiment, adjust the globals
in ```runner.py``` to your setup and run:
```bash
python runner.py --exp cnn_mnist
python plotter.py --exp cnn_mnist
```

Similarly, for the **CIFAR-10 summary** for a version of **ResNet-18** run:
```bash
python runner.py --exp resnet_cifar
python plotter.py --exp resnet_cifar
```
For running the **Kernel Ridge Regression experiment**, you first need to generate the kernel with ```python generate_cntk.py```.
Note: this implementation differs in the kernel choice in ```generate_kernel()``` from the paper. For details on the original
 kernel, please refer to the paper.
 Once you generated the kernel, generate the results by:
 ```bash
python runner.py --exp krr_cifar
python plotter.py --exp krr_cifar 
```

## Continual Learning and Streaming
We showcase the usage our coreset construction in continual learning and streaming with memory replay. 
The buffer regularizer ```beta```  is tuned individually for each method. We provide the best betas 
from ```[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]``` for each method in  ```cl_results/``` and ```streaming_results/```. 

#### Running the Experiments
Change dir to ```cl_streaming```. After this, you can run individual experiments, e.g.: 
```bash
python cl.py --buffer_size 100 --dataset splitmnist --seed 0 --method coreset --beta 100.0
```

You can also run the continual learning and streaming experiments with grid search over ```beta```
on datasets derived from MNIST by adjusting the globals in ```runner.py``` to your setup and running:
```bash
python runner.py --exp cl
python runner.py --exp streaming
python runner.py --exp imbalanced_streaming
```

The table of result can be displayed by running ```python process_results.py``` 
with the corresponding ```--exp``` argument. For example, ```python process_results.py --exp imbalanced_streaming``` 
produces:

| Method \ Dataset  | splitmnistimbalanced   | 
| :-------------: |:-------------:|
| reservoir      | 80.60 +- 4.36 | 
| cbrs      | 89.71 +- 1.31   |  
| coreset | 92.30 +- 0.23   |  

The experiments derived from CIFAR-10 can be similarly run by:
```bash
python cifar_runner.py --exp cl
python process_results --exp splitcifar
python cifar_runner.py --exp imbalanced_streaming
python process_results --exp imbalanced_streaming_cifar
```

## Selection via the Nyström proxy
The Nyström proxy was proposed to support data augmentations. It is also faster for larger coresets than the representer
proxy. An example of running the selection on CIFAR-10 can be found in ```batch_active_learning/nystrom_example.py```.  

## Journal extensions (JMLR 2024)

The `bicoreset/` package implements the parts of *Data Summarization via Bilevel Optimization*
(JMLR 25 (2024) 1-53) that go beyond the conference version. Unlike `bilevel_coreset.py`, which always
selects through a proxy, these run the bilevel construction on the target model itself.

| Paper | Module | What it adds |
| --- | --- | --- |
| Alg. 1, Sec. 3.3 | `bicoreset.direct.BilevelCoreset` | Greedy forward selection with implicit gradients (Eq. (4)/(5)) on any twice differentiable target model |
| Sec. 3.5.1 | `bicoreset.ihvp` | Neumann-series inverse-HVP with loss scaling `alpha`, torch conjugate gradients, identity (GLISTER) approximation, stochastic Hessian |
| Sec. 3.5.1 | `bicoreset.direct` | Binary weights, forward selection in batches, exchange, elimination |
| Alg. 2, Sec. 3.5.3 | `bicoreset.regularized.RegularizedBilevelCoreset` | Weighted coresets via the `L_{1/2}` penalty on the simplex, Duchi projection, adaptive `beta` doubling |
| Sec. 4.4, Eq. (11) | `bicoreset.joint.JointBilevelCoreset` | Joint coresets for several models (alternating and summed) |
| Sec. 4.5, Eq. (12) | `bicoreset.dictionary.DictionarySelector` | Dictionary selection for compressed sensing with L2 / L1 (ISTA) / generative-model recovery |
| Sec. 5.2.1 | `bicoreset.gmm` | Unsupervised coresets for Gaussian mixtures (weighted EM inner solver) |
| Sec. 4.3, Eq. (10) | `batch_active_learning/` | MixMatch training, the bilevel acquisition function, and the baselines of Figure 10 |

Minimal example — Algorithm 1 with binary weights, forward selection in batches and Neumann IHVPs:

```python
import models
from bicoreset import BilevelCoreset, losses

builder = BilevelCoreset(
    model_fn=lambda: models.LogisticRegression(dim, n_classes),
    loss_fn=losses.cross_entropy,          # per-sample losses
    inner_reg=1e-3,
    ihvp='neumann', ihvp_kwargs={'num_terms': 100, 'alpha': 0.05},
    max_inner_it=150, max_outer_it=0)      # max_outer_it=0 => unweighted coreset

coreset_inds, coreset_weights = builder.build(
    X, y, m=100, strategy='forward', selection_batch_size=10)
```

**Note on the loss:** functions passed to `bicoreset` return *per-sample* losses (shape `(n,)`),
not a reduced scalar, so that the inner objective `sum_i w_i l_i(theta)` and the mixed partial
`d^2 f / d theta d w_k = grad_theta l_k` are unambiguous. Ready-made losses live in `bicoreset.losses`.

`max_outer_it > 0` enables the weight optimization of line 6 of Algorithm 1; for deep networks pass a
custom `train_fn(model, X, y, weights)` so the inner problem uses your own schedule and augmentations,
and set `hessian_batch_size` to evaluate Hessian-vector products on a minibatch (Sec. 5.2.3).

`bicoreset/` and `batch_active_learning/` depend only on `numpy` and `torch` (>= 1.11, for `torch.linalg`);
`jax`/`neural-tangents` are needed only if you want the CNTK proxy of `cl_streaming/ntk_generator.py`.

### Demos and tests

```bash
python demos/demo_direct_coreset.py        # forward / batched / exchange / elimination, CG vs Neumann
python demos/demo_bico_reg.py              # Algorithm 2, weighted coresets
python demos/demo_joint_coreset.py         # joint coresets and transferability
python demos/demo_dictionary_selection.py  # compressed sensing measurement selection
python demos/demo_gmm_coreset.py           # unsupervised coresets for GMMs
python demos/demo_active_learning.py       # semi-supervised batch active learning

pytest tests/
```

The tests validate the implementation against the paper's math: the implicit gradient is checked
against finite differences of `G(w)` for a ridge-regression inner problem with a closed-form solution,
the Neumann series against the exact inverse Hessian, and the dictionary-selection gradient against
autograd through the closed-form recovery.

## Citation

If you use the code in a publication, please cite the paper:
```
@article{borsos2020coresets,
      title={Coresets via Bilevel Optimization for Continual Learning and Streaming}, 
      author={Zalán Borsos and Mojmír Mutný and Andreas Krause},
      year={2020},
      journal={arXiv preprint arXiv:2006.03875}
}
```

and, for the extensions in `bicoreset/`:
```
@article{borsos2024data,
      title={Data Summarization via Bilevel Optimization},
      author={Zalán Borsos and Mojmír Mutný and Marco Tagliasacchi and Andreas Krause},
      year={2024},
      journal={Journal of Machine Learning Research},
      volume={25},
      pages={1--53}
}
```
