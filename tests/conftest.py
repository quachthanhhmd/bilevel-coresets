import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _deterministic():
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
