import torch
from torch import Tensor
from typing import Tuple

from .base import Operation


class Multiply2(Operation):

    def __init__(self):
        pass

    def _outputs(self) -> Tuple[Tensor]:
        '''
        Element-wise multiplication
        '''
        assert len(self.inputs) == 2

        return self.inputs[0] * self.inputs[1]

    def _input_grads(self,
                     output_grads: Tuple[Tensor]) -> Tuple[Tensor]:

        return self.inputs[1] * output_grads[0],\
               self.inputs[0] * output_grads[0]


class Add2(Operation):

    def __init__(self):
        pass

    def _outputs(self) -> Tuple[Tensor]:
        '''
        Element-wise addition
        '''
        assert len(self.inputs) == 2

        return self.inputs[0] + self.inputs[1]

    def _input_grads(self, output_grads: Tuple[Tensor]) -> Tuple[Tensor]:

        return output_grads[0], output_grads[0]


class Concat2(Operation):

    def __init__(self):
        pass

    def _outputs(self) -> Tensor:
        '''
        Element-wise multiplication
        '''
        assert len(self.inputs) == 2

        self.input_shapes = [inp.shape[1] for inp in self.inputs]

        return torch.cat(list(self.inputs), dim=1)

    def _input_grads(self,
                     output_grads: Tuple[Tensor]) -> Tuple[Tensor]:
        return torch.split(output_grads[0],
                           self.input_shapes,
                           dim=1)


class Copy(Operation):

    def __init__(self, num=2):
        self.num = num

    def _outputs(self) -> Tuple[Tensor]:
        '''
        Element-wise multiplication
        '''

        output = tuple()
        for i in range(self.num):
            output = output + (self.inputs, )

        return output

    def _input_grads(self, output_grads: Tuple[Tensor]) -> Tensor:
        input_grad = torch.zeros_like(output_grads[0])
        for grad in output_grads:
            input_grad = input_grad + grad
        return input_grad
