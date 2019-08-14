import torch
from torch import Tensor

from typing import Tuple


def permute_data(X: Tensor, y: Tensor, seed=1) -> Tuple[Tensor]:
    perm = torch.randperm(X.shape[0])
    return X[perm], y[perm]


def assert_dim(t: Tensor,
               dim: Tensor):
    assert len(t.shape) == dim, \
        '''
        Tensor expected to have dimension {0}, instead has dimension {1}
        '''.format(dim, len(t.shape))
    return None
