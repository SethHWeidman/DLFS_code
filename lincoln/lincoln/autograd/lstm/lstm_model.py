from typing import List
from lincoln.autograd.layer import Layer
from lincoln.autograd.model import Model
from lincoln.autograd.param import Parameter
from lincoln.autograd.tensor import Tensor


class LSTMModel(Model):
    def __init__(self,
                 layers: List[Layer],
                 vocab_size: int,
                 sequence_length: int = 15,
                 seed: int = 1) -> None:
        super().__init__(layers, seed)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        for layer in self.layers:
            setattr(layer, "seed", self.seed)
            setattr(layer, "vocab_size", self.vocab_size)

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def predict(self, inputs: Tensor) -> Tensor:

        output = Tensor(inputs.data, no_grad=True)

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def parameters(self) -> List[Parameter]:

        params = []
        for layer in self.layers:
            for param in layer.params.values():
                params.append(param)

        return params
