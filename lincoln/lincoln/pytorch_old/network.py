from torch import Tensor
from typing import List

from .layers import Layer, LSTMLayer
from .losses import Loss, MeanSquaredError


class LayerBlock(object):
    '''
    We will ultimately want another level on top of operations and Layers, for example when we get to ResNets.
    For now, I'm calling that a "LayerBlock" and defining a "NeuralNetwork" to be identical to it.
    '''
    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = layers

    def forward(self,
                X_batch: Tensor,
                inference=False) -> Tensor:

        X_out = X_batch
        for layer in self.layers:
            X_out = layer.forward(X_out, inference)

        return X_out

    def backward(self, loss_grad: Tensor) -> Tensor:

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(layer_strs) + ")"


class NeuralNetwork(LayerBlock):
    '''
    Just a list of layers that runs forwards and backwards
    '''
    def __init__(self, layers: List[Layer],
                 loss: Loss = MeanSquaredError):
        super().__init__(layers)
        self.loss = loss

    def forward_loss(self,
                     X_batch: Tensor,
                     y_batch: Tensor) -> float:

        prediction = self.forward(X_batch)
        return self.loss.forward(prediction, y_batch)

    def train_batch(self,
                    X_batch: Tensor,
                    y_batch: Tensor) -> float:

        prediction = self.forward(X_batch)

        batch_loss = self.loss.forward(prediction, y_batch)
        loss_grad = self.loss.backward()

        self.backward(loss_grad)

        return batch_loss


class SequentialNeuralNetwork(NeuralNetwork):

    def __init__(self,
                 num_features: int,
                 sequence_length: int,
                 layers: List[Layer],
                 loss: Loss = MeanSquaredError):
        super().__init__(layers)
        self.loss = loss
        self.num_features = num_features
        self.sequence_length = sequence_length
        for layer in self.layers:
            if issubclass(layer.__class__, LSTMLayer):
                setattr(layer, "vocab_size", self.num_features)
                setattr(layer, "sequence_length", self.sequence_length)

    def forward_loss(self,
                     X_batch: Tensor,
                     y_batch: Tensor) -> float:

        prediction = self.forward(X_batch)
        return self.loss.forward(prediction, y_batch)

    def train_batch(self,
                    X_batch: Tensor,
                    y_batch: Tensor) -> float:

        prediction = self.forward(X_batch)

        batch_loss = self.loss.forward(prediction, y_batch)
        loss_grad = self.loss.backward()

        self.backward(loss_grad)

        return batch_loss
