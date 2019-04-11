import numpy as np
from .base import Operation


class Linear(Operation):
    '''
    Linear activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return self.inputs

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad


class Sigmoid(Operation):
    '''
    Sigmoid activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return 1.0/(1.0+np.exp(-1.0 * self.inputs))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Tanh(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return np.tanh(self.inputs)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        return output_grad * (1 - self.output * self.output)


class ReLU(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return np.clip(self.inputs, 0, None)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        mask = self.output >= 0
        return output_grad * mask
