from typing import Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from lincoln.utils import permute_data
from .model import PyTorchModel
from .preprocessor import PyTorchPreprocessor


class PyTorchTrainer(object):
    def __init__(self,
                 model: PyTorchModel,
                 optim: Optimizer,
                 criterion: _Loss,
                 preprocessor: PyTorchPreprocessor = None):
        self.model = model
        self.optim = optim
        self.loss = criterion
        self.preprocessor = preprocessor
        self._check_optim_net_aligned()

    def _check_optim_net_aligned(self):
        assert self.optim.param_groups[0]['params']\
        == list(self.model.parameters())

    def _generate_batches(self,
                          X: Tensor,
                          y: Tensor,
                          size: int = 32) -> Tuple[Tensor]:

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch


    def fit(self, X_train: Tensor, y_train: Tensor,
            X_test: Tensor, y_test: Tensor,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32):

        for e in range(epochs):
            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self._generate_batches(X_train, y_train,
                                                     batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                if self.preprocessor:
                    X_batch = self.preprocessor.transform(X_batch)

                self.optim.zero_grad()   # zero the gradient buffers
                output = self.model(X_batch)

                loss = self.loss(output, y_batch)
                loss.backward()
                self.optim.step()

            if e % eval_every == 0:
                with torch.no_grad():
                    if self.preprocessor and e == 0:
                        X_test = self.preprocessor.transform(X_test)
                    output = self.model(X_test)
                    loss = self.loss(output, y_test)
                    print(e, loss)
