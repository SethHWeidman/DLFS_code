import numpy as np
from numpy import ndarray

from .base import ParamOperation


class Conv2D(ParamOperation):

    def __init__(self, W: ndarray):
        super().__init__(W)

    def _output(self, inference: bool) -> np.ndarray:
        return np.matmul(self.inputs, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.matmul(output_grad, self.param.transpose(1, 0))

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.matmul(self.inputs.transpose(1, 0), output_grad)


class BiasAdd(ParamOperation):

    def __init__(self,
                 B: np.ndarray):
        super().__init__(B)

    def _output(self, inference: bool) -> np.ndarray:
        return self.inputs + self.param

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.ones_like(self.inputs) * output_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        output_grad_reshape = np.sum(output_grad, axis=0).reshape(1, -1)
        param_grad = np.ones_like(self.param)
        return param_grad * output_grad_reshape
