from torch import Tensor

from .base import Operation


class Flatten(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> Tensor:
        return self.input.view(self.input.shape[0], -1)

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return output_grad.view(self.input.shape)
