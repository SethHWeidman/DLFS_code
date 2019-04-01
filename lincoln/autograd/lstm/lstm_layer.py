import numpy as np
from typing import Dict

from lincoln.autograd.param import Parameter
from lincoln.autograd.tensor import Tensor
from lincoln.autograd.layer import Layer
from lincoln.autograd.activations import sigmoid, tanh


class LSTMLayer(Layer):

    def __init__(self,
                 neurons: int) -> None:
        self.state_size = neurons
        self.first = True
        self.params: Dict[['str'], Tensor] = {}
        self.h_init = Tensor(np.random.randn(1, self.state_size))
        self.c_init = Tensor(np.random.randn(1, self.state_size))

    def _init_params(self, input_: Tensor) -> None:
        np.random.seed(self.seed)

        self.params['Wf'] = Parameter(self.state_size + self.vocab_size,
                                 self.state_size)
        self.params['Wi'] = Parameter(self.state_size + self.vocab_size,
                                 self.state_size)
        self.params['Wo'] = Parameter(self.state_size + self.vocab_size,
                                 self.state_size)
        self.params['Wc'] = Parameter(self.state_size + self.vocab_size,
                                 self.state_size)
        self.params['Wv'] = Parameter(self.state_size, self.vocab_size)

        self.params['Bf'] = Parameter(self.state_size)
        self.params['Bi'] = Parameter(self.state_size)
        self.params['Bo'] = Parameter(self.state_size)
        self.params['Bc'] = Parameter(self.state_size)
        self.params['Bv'] = Parameter(self.vocab_size)

        hiddens = self.h_init.repeat(input_.shape[0])
        cells = self.c_init.repeat(input_.shape[0])

        return hiddens, cells

    def forward(self, input_: Tensor) -> Tensor:
        if self.first:
            self.hiddens, self.cells = self._init_params(input_)
            self.first = False

        for i in range(input_.shape[1]): # sequence length
            if i == 0:
                outputs_single = self._lstm_node(input_.select_index_axis_1(i))
                outputs = outputs_single.expand_dims_axis_1()

            else:
                output_single = self._lstm_node(input_.select_index_axis_1(i))
                output = output_single.expand_dims_axis_1()
                outputs = outputs.append_axis_1(output)

        return outputs

    def _lstm_node(self,
                   inputs: Tensor):

        assert inputs.shape[0] == self.hiddens.shape[0] == self.cells.shape[0]

        Z = inputs.concat(self.hiddens)

        forget = sigmoid(Z @ self.params['Wf'] + self.params['Bf'])

        ingate = sigmoid(Z @ self.params['Wi'] + self.params['Bi'])

        outgate = sigmoid(Z @ self.params['Wo'] + self.params['Bo'])

        change = tanh(Z @ self.params['Wc'] + self.params['Bc'])

        self.cells = self.cells * forget + ingate * change

        self.hiddens = outgate * tanh(self.cells)

        outputs = self.hiddens @ self.params['Wv'] + self.params['Bv']

        return outputs

    def _params(self) -> Tensor:

        return list(self.params.values())
