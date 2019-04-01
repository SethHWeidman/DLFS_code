import torch

from typing import Tuple

from lincoln.autograd.tensor import Tensor
from lincoln.autograd.model import Model
from lincoln.autograd.optim import Optim
from lincoln.utils import permute_data


class Trainer(object):
    '''
    Just a list of layers that runs forwards and backwards
    '''
    def __init__(self,
                 net: Model,
                 optim: Optim) -> None:
        self.net = net
        self.optim = optim

    def update_params(self) -> None:
        self.optim.step(self.net)

    def fit(self, X_train: Tensor, y_train: Tensor,
            X_test: Tensor, y_test: Tensor,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            single_output: bool = False,
            restart: bool = True)-> None:

        if restart:
            self.optim.first = True

        for e in range(epochs):
            torch.manual_seed(seed)
            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self._generate_batches(X_train,
                                                     y_train,
                                                     batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.zero_grad()

                prediction = self.net.predict(X_batch)
                loss = self._loss_prediction(prediction, y_batch)
                loss.backward()

                self.update_params()

            if (e+1) % eval_every == 0:
                predicted = self.net.predict(X_test)
                loss = self._loss_prediction(predicted, y_test)
                print(f"Validation loss after {e+1} epochs is {loss}")

    def _loss_prediction(self,
                         prediction: Tensor,
                         actual: Tensor,
                         kind: str = "mse") -> None:
        if kind == "mse":
            errors = prediction - actual
            loss = (errors * errors).sum()
            return loss

    def _generate_batches(self,
                          X: Tensor,
                          y: Tensor,
                          size: int = 32) -> Tuple[Tensor]:

        if X.shape[0] != y.shape[0]:
            raise ValueError('''feature and label arrays
                             must have the same first dimension''')

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch
