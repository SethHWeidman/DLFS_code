from typing import List

import numpy as np
from numpy import ndarray

from .activations import Linear
from .base import Operation, ParamOperation
from .conv import Conv2D_Op
from .dense import WeightMultiply, BiasAdd
from .dropout import Dropout
from .reshape import Flatten
from lincoln.utils.np_utils import assert_same_shape


class Layer(object):

    def __init__(self,
                 neurons: int) -> None:
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, input_: ndarray) -> None:
        pass

    def forward(self, input_: ndarray,
                inference=False) -> ndarray:

        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)

        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        assert_same_shape(self.input_, input_grad)

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

    def __init__(self,
                 neurons: int,
                 activation: Operation = Linear(),
                 conv_in: bool = False,
                 dropout: float = 1.0,
                 weight_init: str = "standard") -> None:
        super().__init__(neurons)
        self.activation = activation
        self.conv_in = conv_in
        self.dropout = dropout
        self.weight_init = weight_init

    def _setup_layer(self, input_: ndarray) -> None:
        np.random.seed(self.seed)
        num_in = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2/(num_in + self.neurons)
        else:
            scale = 1.0

        # weights
        self.params = []
        self.params.append(np.random.normal(loc=0,
                                            scale=scale,
                                            size=(num_in, self.neurons)))

        # bias
        self.params.append(np.random.normal(loc=0,
                                            scale=scale,
                                            size=(1, self.neurons)))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None


class Conv2D(Layer):
    '''
    Once we define all the Operations and the outline of a layer,
    all that remains to implement here is the _setup_layer function!
    '''
    def __init__(self,
                 out_channels: int,
                 param_size: int,
                 dropout: int = 1.0,
                 weight_init: str = "normal",
                 activation: Operation = Linear(),
                 flatten: bool = False) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.activation = activation
        self.flatten = flatten
        self.dropout = dropout
        self.weight_init = weight_init
        self.out_channels = out_channels

    def _setup_layer(self, input_: ndarray) -> ndarray:

        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2/(in_channels + self.out_channels)
        else:
            scale = 1.0

        conv_param = np.random.normal(loc=0,
                                      scale=scale,
                                      size=(input_.shape[1],  # input channels
                                     self.out_channels,
                                     self.param_size,
                                     self.param_size))

        self.params.append(conv_param)

        self.operations = []
        self.operations.append(Conv2D_Op(conv_param))
        self.operations.append(self.activation)

        if self.flatten:
            self.operations.append(Flatten())

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None
