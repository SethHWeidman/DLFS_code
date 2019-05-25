import torch
from torch import Tensor

from typing import Tuple

from .base import Operation
from ..utils import assert_same_shape


class Activation(Operation):

    def __init__(self):
        pass

    def forward(self, *inputs) -> Tuple[Tensor]:

        assert len(inputs) == 1

        self.inputs = inputs[0]

        self.outputs = self._outputs()

        return self.outputs

    def backward(self, *output_grads) -> Tuple[Tensor]:

        assert len(output_grads) == 1

        output_grads = output_grads[0]

        assert_same_shape(self.outputs, output_grads)

        self.input_grads = self._input_grads(output_grads)

        assert_same_shape(self.inputs, self.input_grads)

        return self.input_grads

    def _outputs(self) -> Tuple[Tensor]:
        raise NotImplementedError()

    def _input_grads(self, output_grad: Tensor) -> Tuple[Tensor]:
        raise NotImplementedError()


class LogSigmoid(Activation):
    def __init__(self):
        super().__init__()

    def _outputs(self) -> Tensor:

        self.outputs = self.inputs - torch.log(torch.exp(self.inputs) + 1)
        return self.outputs

    def _input_grads(self, output_grads: Tensor) -> Tensor:
        return (1 - torch.exp(self.outputs))*output_grads

    def __repr__(self):
        return "LogSigmoid"


class Sigmoid(Activation):
    '''
    Sigmoid activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _outputs(self) -> Tensor:
        return 1.0/(1.0+torch.exp(-1.0 * self.inputs))

    def _input_grads(self, output_grads: Tensor) -> Tensor:
        # Lines specific to this class
        sigmoid_backward = self.outputs * (1.0 - self.outputs)
        input_grads = sigmoid_backward * output_grads
        return input_grads

    def __repr__(self):
        return "Sigmoid"


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def _outputs(self) -> Tensor:
        self.outputs = torch.clamp(self.inputs, 0, 1e5)
        return self.outputs

    def _input_grads(self, output_grads: Tensor) -> Tensor:
        relu_backward = (self.outputs > 0).type(self.outputs.dtype)
        return relu_backward * output_grads

    def __repr__(self):
        return "ReLU"


class LinearAct(Activation):
    '''
    Identity activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _outputs(self) -> Tensor:
        return self.inputs

    def _input_grads(self, output_grads: Tensor) -> Tensor:
        return output_grads


class Tanh(Activation):
    '''
    Tanh activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _outputs(self) -> Tensor:
        return torch.tanh(self.inputs)

    def _input_grads(self, output_grads: Tensor) -> Tensor:
        e = torch.exp(2 * output_grads)
        return (e - 1) / (e + 1)

    def __repr__(self):
        return "Tanh"
