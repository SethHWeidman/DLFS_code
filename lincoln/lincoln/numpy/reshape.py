from numpy import ndarray

from .base import Operation


class Flatten(Operation):
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool = False) -> ndarray:
        return self.input_.reshape(self.input_.shape[0], -1)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad.reshape(self.input_.shape)
