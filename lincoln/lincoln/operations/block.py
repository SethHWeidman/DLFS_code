import torch
from torch import Tensor
from typing import Dict, List, Tuple

from ..utils import assert_same_shapes

# Operations imports
from .base import Operation, ParamOperation
from .fork import Concat2, Copy, Add2, Multiply2
from .activations import Sigmoid, Tanh
from .dense import WeightMultiply, BiasAdd


class OperationBlock(object):

    def __init__(self) -> None:
        self.params: Dict[Tensor] = {}
        self.param_grads: Tuple[Tensor] = []
        self.ops: Dict[Operation] = {}
        self.first: bool = True

    def _setup_block(self) -> Tuple[Tensor]:
        pass

    def forward(self, *inputs) -> Tuple[Tensor]:

        if self.first:
            self._setup_block()
            self.first = False

        self.inputs = inputs

        self.inputs_with_grad = self._inputs_autograd()
        self.params_with_grad = self._params_autograd()
        self._gradify_operations()

        self.outputs = self._outputs()

        return self.outputs

    def _inputs_autograd(self) -> Tuple[Tensor]:
        inputs_with_grad = tuple(inp.detach() for inp in self.inputs)
        for inp in inputs_with_grad:
            inp.requires_grad = True
        return inputs_with_grad

    def _params_autograd(self) -> Tuple[Tensor]:
        params_with_grad = tuple(param.detach()
                                 for param in self.params.values())
        for param in params_with_grad:
            param.requires_grad = True
        return params_with_grad

    def _gradify_operations(self) -> Tuple[Tensor]:
        for op, tensor in zip([op for op in self.ops.values()
                               if issubclass(op.__class__, ParamOperation)],
                              self.params_with_grad):
            setattr(op, "param", tensor)

    def backward(self, *output_grads) -> Tuple[Tensor]:

        assert_same_shapes(self.outputs, output_grads)

        self.input_grads = self._input_grads(output_grads)

        if self.params:
            self.param_grads = self._param_grads()

        assert_same_shapes(self.inputs, self.input_grads)
        return self.input_grads

    def _input_grads(self, output_grads: Tuple[Tensor]) -> Tuple[Tensor]:

        if len(output_grads) == 1:
            self.outputs.backward(output_grads)
        else:
            for out, grad in zip(self.outputs, output_grads):
                out.backward(gradient=grad, retain_graph=True)

        return tuple(x.grad for x in self.inputs_with_grad)

    def _param_grads(self) -> List[Tensor]:
        return tuple(param.grad for param in self.params_with_grad)

    def _params(self) -> None:
        return tuple(param.data for param in self.params_with_grad)

    def _outputs(self) -> Tuple[Tensor]:
        raise NotImplementedError()


class LSTMNode(OperationBlock):

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 seed: int=12345):

        super().__init__()
        self.seed = seed
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(self,
                lstm_params: Dict[str, Tensor],
                *inputs) -> Tuple[Tensor]:

        if self.first:
            self._setup_block(lstm_params)
            self.first = False

        self.inputs = inputs

        self.inputs_with_grad = self._inputs_autograd()

        self.params_with_grad = self._params_autograd()
        self._gradify_operations()

        self.outputs = self._outputs()

        return self.outputs

    def backward(self,
                 lstm_params: Dict[str, Tensor],
                 *output_grads) -> Tuple[Tensor]:

        assert_same_shapes(self.outputs, output_grads)

        self.input_grads = self._input_grads(output_grads)

        if self.params:
            self.param_grads = self._param_grads()

        assert_same_shapes(self.inputs, self.input_grads)
        return self.input_grads

    def _setup_block(self,
                     lstm_params: Dict[str, Tensor]) -> Tuple[Tensor]:

        torch.manual_seed(self.seed)

        self.ops['con'] = Concat2()
        self.ops['copy'] = Copy(4)
        self.ops['sig1'] = Sigmoid()
        self.ops['sig2'] = Sigmoid()
        self.ops['sig3'] = Sigmoid()
        self.ops['tan1'] = Tanh()
        self.ops['tan2'] = Tanh()
        self.ops['mul1'] = Multiply2()
        self.ops['mul2'] = Multiply2()
        self.ops['mul3'] = Multiply2()
        self.ops['add1'] = Add2()
        self.ops['add2'] = Add2()
#         import pdb; pdb.set_trace()
        self.ops['Wf'] = WeightMultiply(lstm_params['Wf'])
        self.ops['Bf'] = BiasAdd(lstm_params['Bf'])

        self.ops['Wi'] = WeightMultiply(lstm_params['Wi'])
        self.ops['Bi'] = BiasAdd(lstm_params['Bi'])

        self.ops['Wc'] = WeightMultiply(lstm_params['Wc'])
        self.ops['Bc'] = BiasAdd(lstm_params['Bc'])

        self.ops['Wo'] = WeightMultiply(lstm_params['Wo'])
        self.ops['Bo'] = BiasAdd(lstm_params['Bo'])

        self.ops['Wv'] = WeightMultiply(lstm_params['Wv'])
        self.ops['Bv'] = BiasAdd(lstm_params['Bv'])

    def _outputs(self) -> Tuple[Tensor]:

        X_in, H_in, C_in = self.inputs_with_grad
        Z = self.ops['con'].forward(X_in, H_in)
        z1, z2, z3, z4 = self.ops['copy'].forward(Z)

        F = self.ops['Wf'].forward(z1)
        F = self.ops['Bf'].forward(F)
        F_out = self.ops['sig1'].forward(F)

        I = self.ops['Wi'].forward(z2)
        I = self.ops['Bi'].forward(I)
        I_out = self.ops['sig2'].forward(I)

        C = self.ops['Wc'].forward(z3)
        C = self.ops['Bc'].forward(C)
        C_bar = self.ops['tan1'].forward(C)

        c1 = self.ops['mul1'].forward(F_out, C_in)
        c2 = self.ops['mul2'].forward(I_out, C_bar)

        c_new = self.ops['add1'].forward(c1, c2)

        O = self.ops['Wo'].forward(z4)
        O = self.ops['Bo'].forward(O)
        O_out = self.ops['sig3'].forward(O)

        C_out = self.ops['tan2'].forward(c_new)

        H_out = self.ops['mul3'].forward(O_out, C_out)

        X = self.ops['Wv'].forward(H_out)
        X_out = self.ops['Bv'].forward(X)

        return X_out, H_out, C_out
