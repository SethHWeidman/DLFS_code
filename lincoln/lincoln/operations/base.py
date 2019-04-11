from torch import Tensor, nn
from typing import Tuple

from ..utils import assert_same_shapes


class Operation(object):

    def __init__(self):
        pass

    def forward(self, *inputs) -> Tuple[Tensor]:

        if len(inputs) == 1:
            self.inputs = inputs[0]
        else:
            self.inputs = inputs

        self.outputs = self._outputs()

        return self.outputs

    def backward(self, *output_grads) -> Tuple[Tensor]:

        assert_same_shapes(self.outputs, output_grads)

        self._input_grads(output_grads)

        assert_same_shapes(self.inputs, self.input_grads)

        return self.input_grads

    def _outputs(self) -> Tuple[Tensor]:
        raise NotImplementedError()

    def _input_grads(self, *output_grads) -> Tuple[Tensor]:
        raise NotImplementedError()


class ParamOperation(Operation):

    def __init__(self, param: Tensor) -> Tensor:
        super().__init__()
        self.param = param

    def backward(self, *output_grads) -> Tuple[Tensor]:

        # import pdb; pdb.set_trace()
        if len(output_grads) == 1:
            output_grads = output_grads[0]

        assert_same_shapes(self.outputs, output_grads)

        self.input_grads = self._input_grads(output_grads)
        self.param_grad = self._param_grad(output_grads)

        assert_same_shapes(self.inputs, self.input_grads)

        return self.input_grads

    def _param_grad(self, *output_grad) -> Tensor:
        raise NotImplementedError()


class PyTorchOperation(ParamOperation):

    def __init__(self, param: Tensor) -> Tensor:
        super().__init__(param)
        self.op = nn.Linear(param.shape[0],
                            param.shape[0])

    def _output(self) -> Tensor:

        self.input_with_grad = self.input.detach()
        self.input_with_grad.requires_grad = True

        return self.op(self.input_with_grad)

    def _input_grad(self, output_grad: Tensor) -> Tensor:

        self.output.backward(gradient=output_grad)
        return self.input_with_grad.grad

    def _param_grad(self, output_grad: Tensor) -> Tensor:
        return self.op.weight.grad
