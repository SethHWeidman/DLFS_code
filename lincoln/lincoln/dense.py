import numpy as np

from lincoln import base


class WeightMultiply(base.ParamOperation):

    def __init__(self, W: np.ndarray):
        super().__init__(W)

    def _output(self) -> np.ndarray:
        return np.matmul(self.input_, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.matmul(output_grad, self.param.transpose(1, 0))

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.matmul(self.input_.transpose(1, 0), output_grad)


class BiasAdd(base.ParamOperation):

    def __init__(self, B: np.ndarray):
        super().__init__(B)

    def _output(self) -> np.ndarray:
        return self.input_ + self.param

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        output_grad_reshape = np.sum(output_grad, axis=0).reshape(1, -1)
        param_grad = np.ones_like(self.param)
        return param_grad * output_grad_reshape
