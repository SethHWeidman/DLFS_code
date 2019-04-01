import torch
from torch import Tensor

from typing import Tuple

from .base import ParamOperation


class WeightMultiply(ParamOperation):

    def __init__(self, W: Tensor):
        super().__init__(W)

    def _outputs(self) -> Tuple[Tensor]:
        return torch.mm(self.inputs, self.param)

    def _input_grads(self, *output_grads) -> Tensor:
        return torch.mm(output_grads[0], self.param.transpose(0, 1))

    def _param_grad(self, *output_grads) -> Tensor:
        return torch.mm(self.inputs.transpose(0, 1), output_grads[0])


class BiasAdd(ParamOperation):

    def __init__(self,
                 B: Tensor):
        super().__init__(B)

    def _outputs(self) -> Tensor:
        return torch.add(self.inputs, self.param)

    def _input_grads(self, *output_grads) -> Tensor:
        return torch.ones_like(self.inputs[0]) * output_grads[0]

    def _param_grad(self, *output_grads) -> Tensor:
        param_grad = torch.ones_like(self.param) * output_grads[0]
        return torch.sum(param_grad, dim=0).reshape(1, param_grad.shape[1])
