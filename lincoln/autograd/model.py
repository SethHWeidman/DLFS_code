import inspect
from typing import Iterator, List
from lincoln.autograd.tensor import Tensor
from lincoln.autograd.param import Parameter


class Model:
    def __init__(self,
                 layers,
                 seed: int = 1) -> None:
        self.seed = seed
        self.layers = layers

        for layer in self.layers:
            setattr(layer, "seed", self.seed)


    # def parameters(self) -> Iterator[Parameter]:
    #     for name, value in inspect.getmembers(self):
    #         if isinstance(value, Parameter):
    #             yield value
    #         elif isinstance(value, Model):
    #             yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def predict(self, inputs: Tensor) -> Tensor:

        output = inputs

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def parameters(self) -> List[Parameter]:

        params = []
        for layer in self.layers:
            for param in layer.params.values():
                params.append(param)

        return params
