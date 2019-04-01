
from typing import List, NamedTuple, Callable, Optional, Union

import numpy as np

Arrayable = Union[float, list, np.ndarray]

Tensorable = Union['Tensor', float, np.ndarray]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

def collapse_sum(grad: np.ndarray,
                 t: 'Tensor') -> np.ndarray:

    # Sum out added dims
    # import pdb; pdb.set_trace()
    ndims_added = grad.ndim - t.data.ndim
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)

    # Sum across broadcasted (but non-added dims)
    for i, dim in enumerate(t.shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor:

    def __init__(self,
                 data: np.ndarray,
                 depends_on: List[Dependency] = None,
                 no_grad: bool = False) -> None:
        self.data = ensure_array(data)
        self.depends_on = depends_on or []
        self.no_grad = no_grad
        self.shape = self.data.shape
        self.grad: Optional[np.ndarray] = None
        if not self.no_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        return f"Tensor({np.round(self.data, 4)})"

    def __add__(self, other: Tensorable) -> 'Tensor':
        """gets called if I do t + other"""
        return _add(self, ensure_tensor(other))

    def __radd__(self, other: Tensorable) -> 'Tensor':
        """gets called if I do other + t"""
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other: Tensorable) -> 'Tensor':
        """when we do t += other"""
        self.data = self.data + ensure_tensor(other).data
        return self

    def __isub__(self, other: Tensorable) -> 'Tensor':
        """when we do t -= other"""
        self.data = self.data - ensure_tensor(other).data
        return self

    def __imul__(self, other: Tensorable) -> 'Tensor':
        """when we do t *= other"""
        self.data = self.data * ensure_tensor(other).data
        return self

    def __mul__(self, other: Tensorable) -> 'Tensor':
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other: Tensorable) -> 'Tensor':
        return _mul(ensure_tensor(other), self)

    def __matmul__(self, other: Tensorable) -> 'Tensor':
        return _matmul(self, other)

    def __neg__(self) -> 'Tensor':
        return _neg(self)

    def __sub__(self, other: Tensorable) -> 'Tensor':
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other: Tensorable) -> 'Tensor':
        return _sub(ensure_tensor(other), self)

    def __getitem__(self, idxs) -> 'Tensor':
        return _slice(self, idxs)

    def concat(self, other: Tensorable) -> 'Tensor':
        return _concat(self, ensure_tensor(other))

    def repeat(self, repeats: int) -> 'Tensor':
        return _repeat(self, repeats)

    def mean_axis_0(self) -> 'Tensor':
        return _mean_axis_0(self)

    def expand_dims_axis_1(self) -> 'Tensor':
        return _expand_dims_axis_1(self)

    def append_axis_1(self, other: Tensorable) -> 'Tensor':
        return _append_axis_1(self, ensure_tensor(other))

    def select_index_axis_1(self, ind: int) -> 'Tensor':
        return _select_index_axis_1(self, ind)

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64),
                           no_grad = True)

    def backward(self, grad: 'Tensor' = None) -> None:
        if self.no_grad:
            return

        if self.shape == ():
            grad = np.array(1.0)

        self.grad = self.grad + grad

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad)
            dependency.tensor.backward(backward_grad)

    def sum(self) -> 'Tensor':
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:

    def _forward(t: Tensor):
        return t.data.sum()

    def t_grad(grad: np.ndarray) -> np.ndarray:
        return grad * np.ones_like(t.data)

    data = _forward(t)
    depends_on = [
        Dependency(t, t_grad),
    ]

    return Tensor(data, depends_on)


########### SPECIAL GRAD FUNCTIONS ##################

def _add(t1: Tensor, t2: Tensor) -> Tensor:

    def _forward(t1: Tensor, t2: Tensor) -> np.ndarray:
        return t1.data + t2.data

    def t1_grad(grad: np.ndarray) -> np.ndarray:

        grad = collapse_sum(grad, t1)

        return grad

    def t2_grad(grad: np.ndarray) -> np.ndarray:

        grad = collapse_sum(grad, t2)

        return grad

    data = _forward(t1, t2)
    depends_on = [
        Dependency(t1, t1_grad),
        Dependency(t2, t2_grad)
    ]
    return Tensor(data, depends_on)

def _mul(t1: Tensor, t2: Tensor) -> Tensor:

    def _forward(t1: Tensor, t2: Tensor) -> np.ndarray:
        return t1.data * t2.data

    def t1_grad(grad: np.ndarray) -> np.ndarray:
        grad = grad * t2.data
        grad = collapse_sum(grad, t1)

        return grad

    def t2_grad(grad: np.ndarray) -> np.ndarray:
        grad = grad * t1.data
        grad = collapse_sum(grad, t2)

        return grad

    data = _forward(t1, t2)
    depends_on = [
        Dependency(t1, t1_grad),
        Dependency(t2, t2_grad)
    ]
    return Tensor(data, depends_on)

def _matmul(t1: Tensor, t2: Tensor) -> Tensor:

    assert t1.shape[1] == t2.shape[0]

    def _forward(t1: Tensor, t2: Tensor) -> np.ndarray:
        return t1.data @ t2.data

    def t1_grad(grad: np.ndarray) -> np.ndarray:
        grad = grad @ t2.data.T

        return grad

    def t2_grad(grad: np.ndarray) -> np.ndarray:
        grad = t1.data.T @ grad

        return grad

    data = _forward(t1, t2)
    depends_on = [
        Dependency(t1, t1_grad),
        Dependency(t2, t2_grad)
    ]
    return Tensor(data, depends_on)

def _neg(t: Tensor) -> Tensor:

    def _forward(t: Tensor) -> np.ndarray:
        return -t.data

    def t_grad(grad: np.ndarray) -> np.ndarray:
        return -grad

    data = _forward(t)
    depends_on = [
        Dependency(t, t_grad),
    ]
    return Tensor(data, depends_on)

def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2

def _slice(t: Tensor, idxs: slice) -> Tensor:

    def _forward(t: Tensor, idxs):
        return t.data[idxs]

    data = _forward(t, idxs)

    def t_grad(grad: np.ndarray) -> np.ndarray:
        bigger_grad = np.zeros_like(t.data)
        bigger_grad[idxs] = grad
        return bigger_grad

    depends_on = [
        Dependency(t, t_grad),
    ]

    return Tensor(data, depends_on, t.no_grad)

def _concat(t1: Tensor, t2: Tensor) -> Tensor:

    assert t1.shape[0] == t2.shape[0],\
    "Concatenated Tensors must have the same shape along first dimension"

    def _forward(t1: Tensor, t2: Tensor):
        return np.concatenate([t1.data, t2.data], axis=1)

    def t1_grad(grad: np.ndarray) -> np.ndarray:
        return grad[:,:t1.shape[1]]

    def t2_grad(grad: np.ndarray) -> np.ndarray:
        return grad[:,t1.shape[1]:]

    data = _forward(t1, t2)

    depends_on = [
        Dependency(t1, t1_grad),
        Dependency(t2, t2_grad)
    ]

    return Tensor(data, depends_on)

def _repeat_axis_0(t: Tensor, repeats: int) -> Tensor:

    assert t.shape[0] == 1,\
    "Repeat operation should only be used on rows"

    def _forward(t: Tensor, repeats: int) -> np.ndarray:
        return np.repeat(t.data, repeats, axis=0)

    def t_grad(grad: np.ndarray) -> np.ndarray:
        return grad.sum(axis=0)

    data = _forward(t, repeats)

    depends_on = [
        Dependency(t, t_grad)
    ]

    return Tensor(data, depends_on)

def _expand_dims_axis_1(t: Tensor) -> Tensor:

    assert t.data.ndim == 2

    def _forward(t: Tensor) -> np.ndarray:
        return np.expand_dims(t.data, axis=1)

    def t_grad(grad: np.ndarray) -> np.ndarray:

        assert grad.ndim == 3

        return grad[:, 0, :]

    data = _forward(t)

    depends_on = [
        Dependency(t, t_grad)
    ]

    return Tensor(data, depends_on)

def _append_axis_1(t1: Tensor, t2: Tensor) -> Tensor:

    assert t1.data.ndim == t2.data.ndim == 3

    def _forward(t1: Tensor, t2: Tensor) -> np.ndarray:
        return np.append(t1.data, t2.data, axis=1)

    def t1_grad(grad: np.ndarray) -> np.ndarray:

        assert grad.ndim == 3

        return grad[:, :t1.shape[1], :]

    def t2_grad(grad: np.ndarray) -> np.ndarray:

        assert grad.ndim == 3

        return grad[:, t1.shape[1]:, :]

    data = _forward(t1, t2)

    depends_on = [
        Dependency(t1, t1_grad),
        Dependency(t2, t2_grad)
    ]

    return Tensor(data, depends_on)

def _select_index_axis_1(t: Tensor, ind: int) -> Tensor:

    assert t.data.ndim == 3

    def _forward(t: Tensor, ind: int) -> np.ndarray:
        return t.data[:, ind, :]

    data = _forward(t, ind)

    def t_grad(grad: np.ndarray) -> np.ndarray:

        assert grad.ndim == 2

        bigger_grad = np.zeros_like(t.data)
        bigger_grad[:, ind, :] = grad
        return bigger_grad

    depends_on = [
        Dependency(t, t_grad)
    ]

    return Tensor(data, depends_on, t.no_grad)


def _mean_axis_0(t: Tensor) -> Tensor:

    assert len(t.shape) == 2,\
    "Mean_axis_0 should only be used on 2d Tensors"

    batch_size = t.shape[0]

    def _forward(t: Tensor) -> np.ndarray:
        return t.data.mean(axis=0)

    def t_grad(grad: np.ndarray) -> np.ndarray:
        return grad.repeat(batch_size, axis=0) / batch_size

    data = _forward(t)

    depends_on = [
        Dependency(t, t_grad)
    ]

    return Tensor(data, depends_on)
