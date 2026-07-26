"""Semi-supervised batch active learning via bilevel coresets (Sec. 4.3 / 5.5)."""

from batch_active_learning.mixmatch import MixMatchTrainer, sharpen, mixup
from batch_active_learning.proxy import (
    NystromFeatureMap,
    make_rbf_kernel,
    nystrom_feature_map,
)
from batch_active_learning.acquisition import (
    ACQUISITION_FNS,
    badge,
    bico,
    consistency,
    get_acquisition_fn,
    kcenter,
    max_entropy,
    uniform,
)
from batch_active_learning.active_learning import ActiveLearningLoop

__all__ = [
    'MixMatchTrainer', 'sharpen', 'mixup',
    'NystromFeatureMap', 'make_rbf_kernel', 'nystrom_feature_map',
    'ACQUISITION_FNS', 'get_acquisition_fn',
    'uniform', 'max_entropy', 'kcenter', 'consistency', 'badge', 'bico',
    'ActiveLearningLoop',
]
