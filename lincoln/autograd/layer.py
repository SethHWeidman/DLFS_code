import numpy as np

from typing import Dict, Callable

from lincoln.autograd.tensor import Tensor
from lincoln.autograd.param import Parameter
from lincoln.autograd.activations import linear


class Layer(object):

    def __init__(self,
                 neurons: int,
                 activation: Callable[[Tensor], Tensor] = linear) -> None:
        self.num_hidden = neurons
        self.activation = activation
        self.first = True
        self.params: Dict[['str'], Tensor] = {}

    def _init_params(self, input_: Tensor) -> None:
        np.random.seed(self.seed)
        pass

    def forward(self, input_: Tensor) -> Tensor:
        if self.first:
            self._init_params(input_)
            self.first = False

        output = self._output(input_)

        return output

    def _params(self) -> Tensor:
        return [self.params.values()]

    def _output(self, input_: Tensor) -> Tensor:
        raise NotImplementedError()


class Dense(Layer):

    def __init__(self,
                 neurons: int,
                 activation: Callable[[Tensor], Tensor] = linear) -> None:
        super().__init__(neurons, activation)

    def _init_params(self, input_: Tensor) -> None:
        np.random.seed(self.seed)
        self.params['W'] = Parameter(input_.shape[1], self.num_hidden)
        self.params['B'] = Parameter(self.num_hidden)

    def _output(self, input_: Tensor) -> Tensor:

        neurons = input_ @ self.params['W'] + self.params['B']

        return self.activation(neurons)
