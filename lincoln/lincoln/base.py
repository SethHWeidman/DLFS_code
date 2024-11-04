import numpy as np

from lincoln.utils import np_utils


class Operation(object):

    def __init__(self):
        pass

    def forward(self, input_: np.ndarray) -> np.ndarray:

        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        np_utils.assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        np_utils.assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self) -> np.ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ParamOperation(Operation):

    def __init__(self, param: np.ndarray) -> np.ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        np_utils.assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        np_utils.assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
