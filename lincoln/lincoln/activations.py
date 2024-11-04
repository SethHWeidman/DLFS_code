import numpy as np
from lincoln import base


class Linear(base.Operation):
    """
    Linear activation function
    """

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad


class Sigmoid(base.Operation):
    """
    Sigmoid activation function
    """

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Tanh(base.Operation):
    """
    Hyperbolic tangent activation function
    """

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        return output_grad * (1 - self.output * self.output)


class ReLU(base.Operation):
    """
    Hyperbolic tangent activation function
    """

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        return np.clip(self.input_, 0, None)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        mask = self.output >= 0
        return output_grad * mask
