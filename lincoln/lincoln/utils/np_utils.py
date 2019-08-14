from typing import Tuple
import numpy as np
from scipy.special import logsumexp


def to_2d(a: np.ndarray,
          type: str="col") -> np.ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
        "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)


def normalize(a: np.ndarray):
    other = 1 - a
    return np.concatenate([a, other], axis=1)


def unnormalize(a: np.ndarray):
    return a[np.newaxis, 0]


def permute_data(X: np.ndarray, y: np.ndarray):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


Batch = Tuple[np.ndarray, np.ndarray]


def generate_batch(X: np.ndarray,
                   y: np.ndarray,
                   start: int = 0,
                   batch_size: int = 10) -> Batch:

    assert (X.dim() == 2) and (y.dim() == 2), \
        "X and Y must be 2 dimensional"

    if start+batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start:start+batch_size], y[start:start+batch_size]

    return X_batch, y_batch


def assert_same_shape(output: np.ndarray,
                      output_grad: np.ndarray):
    assert output.shape == output_grad.shape, \
        '''
        Two tensors should have the same shape;
        instead, first Tensor's shape is {0}
        and second Tensor's shape is {1}.
        '''.format(tuple(output_grad.shape), tuple(output.shape))
    return None


def assert_dim(t: np.ndarray,
               dim: int):
    assert t.ndim == dim, \
        '''
        Tensor expected to have dimension {0}, instead has dimension {1}
        '''.format(dim, len(t.shape))
    return None


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))
