import numpy as np

from lincoln.autograd.tensor import Tensor, Dependency

def sigmoid(t: Tensor) -> Tensor:

    def _forward(t: Tensor) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(t.data)))

    data = _forward(t)

    def t_grad(grad: np.ndarray) -> np.ndarray:
        return grad * data * (1.0 - data)

    depends_on = [
        Dependency(t, t_grad)
    ]

    return Tensor(data, depends_on)


def tanh(t: Tensor) -> Tensor:
    def _forward(t: Tensor):
        return np.tanh(t.data)

    data = _forward(t)

    def t_grad(grad: np.ndarray) -> np.ndarray:
        return grad * (1 - data * data)

    depends_on = [
        Dependency(t, t_grad)
    ]

    return Tensor(data, depends_on)


def linear(t: Tensor) -> Tensor:
    def _forward(t: Tensor):
        return t.data

    data = _forward(t)

    def t_grad(grad: np.ndarray) -> np.ndarray:
        return grad

    depends_on = [
        Dependency(t, t_grad)
    ]

    return Tensor(data, depends_on)
