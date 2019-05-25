from typing import List, Tuple, Dict

import torch
from torch import Tensor

from .operations.block import LSTMNode
from .operations.activations import Sigmoid
from .operations.base import Operation, ParamOperation
from .operations.dense import WeightMultiply, BiasAdd
from .operations.conv import Conv2D_Op, Conv2D_Op_cy, Conv2D_Op_Pyt
from .operations.reshape import Flatten
from .utils import assert_same_shapes


class Layer(object):

    def __init__(self,
                 neurons: int) -> None:
        self.neurons = neurons
        self.first = True
        self.params: List[Tensor] = []
        self.param_grads: List[Tensor] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, input_: Tensor) -> None:
        pass

    def forward(self, input_: Tensor,
                inference=False) -> Tensor:
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: Tensor) -> Tensor:

        assert_same_shapes(self.output, output_grad)

        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        assert_same_shapes(self.input_, input_grad)

        self._param_grads()

        return input_grad

    def _param_grads(self) -> None:

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> None:

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    '''
    Once we define all the Operations and the outline of a layer, all that remains to implement here
    is the _setup_layer function!
    '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid(),
                 conv_in: bool = False) -> None:
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: Tensor) -> None:

        # weights
        self.params.append(torch.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(torch.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None


class Conv2D(Layer):
    '''
    Once we define all the Operations and the outline of a layer,
    all that remains to implement here is the _setup_layer function!
    '''
    def __init__(self,
                 out_channels: int,
                 param_size: int,
                 activation: Operation = Sigmoid(),
                 cython: bool = False,
                 pytorch: bool = False,
                 flatten: bool = False) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.activation = activation
        self.cython = cython
        self.pytorch = pytorch
        self.flatten = flatten
        self.neurons = out_channels

    def _setup_layer(self, input_: Tensor) -> Tensor:

        conv_param = torch.empty(self.neurons,
                                 input_.shape[1],
                                 self.param_size,
                                 self.param_size).uniform_(-1, 1)
        self.params.append(conv_param)

        self.operations = []

        if self.pytorch:
            self.operations.append(Conv2D_Op_Pyt(self.params_dict[0]))
        elif self.cython:
            self.operations.append(Conv2D_Op_cy(self.params_dict[0]))
        else:
            self.operations.append(Conv2D_Op(self.params_dict[0]))

        self.operations.append(self.activation)

        if self.flatten:
            self.operations.append(Flatten())

        return None

class BatchNorm(Layer):

    def __init__(self) -> None:
        pass

    def _setup_layer(self, input_: Tensor) -> None:
        obs = input_[0]

        self.aggregates = (0,
                           np.zeros_like(obs),
                           np.zeros_like(obs))

        self.params: List[float] = []
        self.params.append(0.)
        self.params.append(1.)

    def _update_stats(self, new_input: Tensor):

        (count, mean, M2) = self.aggregates
        count += 1
        delta = new_input - mean
        mean += delta / count
        delta2 = new_input - mean
        M2 += delta * delta2

        self.aggregates = (count, mean, M2)


    def forward(self, input_: Tensor,
                inference=False) -> Tensor:

        self.input_ = input_
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        if not inference:
            for obs in input_:
                self._update_stats(obs)

            self.mean = input_.mean(axis=0)
            self.var = input_.var(axis=0)
        else:
            self.mean, self.var, samp_var = finalize(self.aggregates)

        self.output = (input_ - self.mean) / (self.var + 1e-8)

        self.output *= self.params[0] # gamma
        self.output += self.params[0] # beta

        return self.output

    def backward(self,
                 output_grad: Tensor) -> Tensor:

        assert_same_shape(self.output, output_grad)

        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        dbeta = np.sum(output_grad, axis=0)
        dgamma = np.sum((self.input_ - mu) * \
                        np.sqrt((self.var + 1e-8)) * output_grad, axis=0)

        self.param_grads = [dbeta, dgamma]

        input_grad = (self.params[1] * np.sqrt(self.var + 1e-8) / N) * \
                     (N * output_grad - np.sum(output_grad, axis=0) - \
                      (self.input_ - self.mean) * (self.var + 1e-8)**(-1.0) * \
                      np.sum(output_grad * (input_ - self.mean), axis=0))

        assert_same_shape(self.input_, input_grad)

        return input_grad

# LSTMLayer class - series of operations
class LSTMLayer(object):

    def __init__(self,
                 neurons: int = 100,
                 weight_scale: float = 0.01):
        super().__init__()

        self.hidden_size = neurons
        self.vocab_size: int = None
        self.sequence_length: int = None
        self.first: bool = True
        self.start_H: Tensor = None
        self.start_C: Tensor = None
        self.params_dict: Dict[Tensor] = {}
        self.weight_scale = weight_scale

    def _init_nodes(self):
        self.nodes = [LSTMNode(self.hidden_size, self.vocab_size)
                      for _ in range(self.sequence_length)]

    def _init_params(self, input_: Tensor) -> Tensor:
        '''
        First dimension of input_ will be batch size
        '''
        self._init_nodes()

        self.start_H = torch.zeros(input_.shape[0], self.hidden_size)
        self.start_C = torch.zeros(input_.shape[0], self.hidden_size)
        self.params_dict['Wf'] = torch.randn(self.hidden_size + self.vocab_size, self.hidden_size)
        self.params_dict['Bf'] = torch.randn(1, self.hidden_size)

        self.params_dict['Wi'] = torch.randn(self.hidden_size + self.vocab_size, self.hidden_size)
        self.params_dict['Bi'] = torch.randn(1, self.hidden_size)

        self.params_dict['Wc'] = torch.randn(self.hidden_size + self.vocab_size, self.hidden_size)
        self.params_dict['Bc'] = torch.randn(1, self.hidden_size)

        self.params_dict['Wo'] = torch.randn(self.hidden_size + self.vocab_size, self.hidden_size)
        self.params_dict['Bo'] = torch.randn(1, self.hidden_size)

        self.params_dict['Wv'] = torch.randn(self.hidden_size,
                                       self.vocab_size)
        self.params_dict['Bv'] = torch.randn(1, self.vocab_size)

        # for param in self.params_dict.values():
        #     param.mul_(self.weight_scale)

        for param in self.params_dict.values():
            param.requires_grad = True

    def _zero_param_grads(self) -> None:
        for param in self.params_dict.values():
            if param.grad is not None:
                param.grad.data.zero_()

    def _params(self) -> None:
        return tuple(self.params_dict.values())

    def _param_grads(self) -> None:
        return tuple(param.grad for param in self.params_dict.values())

    def _clip_gradients(self) -> None:
        for grad in self.param_grads:
            grad.data = torch.clamp(grad.data, -2, 2)

    def forward(self, input_: Tensor) -> Tensor:

        if self.first:
            self._init_params(input_)
            self.first = False

        # shape: batch size by sequence length by vocab_size
        self.input_ = input_

        batch_size = self.input_.shape[0]

        H_in = torch.clone(self.start_H.expand(batch_size,
                                               self.hidden_size))
        C_in = torch.clone(self.start_C.expand(batch_size,
                                               self.hidden_size))

        self.output = torch.zeros_like(self.input_)

        seq_len = self.input_.shape[1]
        for i in range(seq_len):

            # pass info forward through the nodes
            elem_out, H_in, C_in = self.nodes[i].\
                forward(self.params_dict, self.input_[:, i, :], H_in, C_in)

            self.output[:, i, :] = elem_out

        self.start_H = H_in.mean(dim=0)
        self.start_C = C_in.mean(dim=0)

        return self.output

    def backward(self, output_grad: Tensor) -> Tensor:

        self._zero_param_grads()

        batch_size = output_grad.shape[0]

        dH_next = torch.zeros_like(self.start_H.expand(batch_size,
                                                       self.hidden_size))
        dC_next = torch.zeros_like(self.start_C.expand(batch_size,
                                                       self.hidden_size))

        self.input_grad = torch.zeros_like(self.input_)

        for i in reversed(range(self.input_.shape[1])):

            # pass info forward through the nodes
            grad_out, dH_next, dC_next = \
                self.nodes[i].backward(self.params_dict, output_grad[:, i, :],
                                       dH_next, dC_next)

            self.input_grad[:, i, :] = grad_out

        self.params = self._params()
        self.param_grads = self._param_grads()
        self._clip_gradients()

        return self.input_grad
