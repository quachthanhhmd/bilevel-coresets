"""Bilevel coreset construction directly on target models.

Reference implementation of the extensions introduced in

    Borsos, Mutny, Tagliasacchi, Krause.
    "Data Summarization via Bilevel Optimization", JMLR 25 (2024) 1-53.

The original repository (Borsos et al., NeurIPS 2020) only implements coreset
selection *via a proxy* (``bilevel_coreset.BilevelCoreset``).  This package adds
the parts of the journal version that operate directly on the target model:

===========================  =====================================  ==========================
Component                    Paper reference                        Module
===========================  =====================================  ==========================
Algorithm 1 (BiCo)           Sec. 3.3, Eq. (1)-(5)                  :mod:`bicoreset.direct`
Binary weights / IHVP        Sec. 3.5.1                             :mod:`bicoreset.ihvp`
Forward-in-batches /         Sec. 3.5.1                             :mod:`bicoreset.direct`
exchange / elimination
Algorithm 2 (BiCo Reg)       Sec. 3.5.3, Eq. (8)-(9)                :mod:`bicoreset.regularized`
Joint coresets               Sec. 4.4, Eq. (11)                     :mod:`bicoreset.joint`
Dictionary selection         Sec. 4.5, Eq. (12)                     :mod:`bicoreset.dictionary`
GMM (unsupervised) coresets  Sec. 5.2.1                             :mod:`bicoreset.gmm`
Batch active learning        Sec. 4.3, Eq. (10)                     ``batch_active_learning/``
===========================  =====================================  ==========================
"""

from bicoreset.direct import BilevelCoreset
from bicoreset.regularized import RegularizedBilevelCoreset
from bicoreset.joint import JointBilevelCoreset
from bicoreset.dictionary import DictionarySelector
from bicoreset.gmm import GMMCoreset, WeightedGMM
from bicoreset.ihvp import (
    CGInverseHVP,
    IdentityInverseHVP,
    NeumannInverseHVP,
    create_ihvp,
    hvp,
)
from bicoreset import losses

__all__ = [
    'BilevelCoreset',
    'RegularizedBilevelCoreset',
    'JointBilevelCoreset',
    'DictionarySelector',
    'GMMCoreset',
    'WeightedGMM',
    'CGInverseHVP',
    'IdentityInverseHVP',
    'NeumannInverseHVP',
    'create_ihvp',
    'hvp',
    'losses',
]
