
from typing import Callable, Tuple, List
from copy import deepcopy

import numpy as np
import torch
from torch import Tensor

from .network import NeuralNetwork
from .optimizers import Optimizer
from ..np_utils import permute_data, to_2d


class Trainer(object):
    '''
    Just a list of layers that runs forwards and backwards
    '''
    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer) -> None:
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def update_params(self) -> None:
        self.optim.step()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            single_output: bool = False,
            restart: bool = True,
            early_stopping: bool = True)-> None:

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):

            if (e+1) % eval_every == 0:

                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train,
                                                    batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

            if (e+1) % eval_every == 0:

                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)

                if early_stopping:
                    if loss < self.best_loss:
                        print(f"Validation loss after {e+1} epochs is {loss:.3f}")
                        self.best_loss = loss
                    else:
                        print(f"""Loss increased after epoch {e+1}, final loss was {self.best_loss:.3f}, using the model from epoch {e+1-eval_every}""")
                        self.net = last_model
                        # ensure self.optim is still updating self.net
                        setattr(self.optim, 'net', self.net)
                        break
                else:
                    print(f"Validation loss after {e+1} epochs is {loss:.3f}")

            if self.optim.decay_lr:
                self.optim._decay_lr(epoch=e+1,
                                     max_epochs=epochs)


    def generate_batches(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         size: int = 32) -> Tuple[Tensor]:

        assert X.shape[0] == y.shape[0], \
        '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(X.shape[0], y.shape[0])

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch


class LSTMTrainer(Trainer):

    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer,
                 data: str) -> None:
        super().__init__(net, optim)
        self.data = data
        self.train_data, self.test_data = self._train_test_split_text()

        self.vocab_size = self.net.num_features
        self.max_len = self.net.sequence_length

        self.chars = list(set(self.data))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def fit(self,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            single_output: bool = False,
            restart: bool = True)-> None:

        if restart:
            self.optim.first = True

        torch.manual_seed(seed)

        for e in range(epochs):

            batch_generator = self.generate_batches_next_char(batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                print(self.net.train_batch(X_batch, y_batch))
                print(ii)
                if np.isnan(self.net.train_batch(X_batch, y_batch)):
                    import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
                # self.net.layers[0].params_dict['Wf'][1:3, 1:3]
                self.update_params()

            if (e+1) % eval_every == 0:

                X_test, y_test = self.generate_test_data()

                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)
                print(f"Validation loss after {e+1} epochs is {loss:.3f}")

    def _train_test_split_text(self, pct=0.8) -> Tuple[str]:

        n = len(self.data)
        return self.data[:int(n * pct)], self.data[int(n * pct):]

    def generate_batches_next_char(self,
                                   batch_size: int) -> Tuple[Tensor]:
        N = len(self.train_data)
        # add batch size
        for ii in range(0, N, batch_size):

            features_tensors = []
            target_tensors = []

            for char in range(batch_size):

                features_str, target_str =\
                 self.train_data[ii+char:ii+char+self.max_len],\
                 self.train_data[ii+char+1:ii+char+self.max_len+1]

                features_array, target_array =\
                    self._string_to_one_hot_array(features_str),\
                    self._string_to_one_hot_array(target_str)

                features_tensors.append(features_array)
                target_tensors.append(target_array)

            yield torch.stack(features_tensors), torch.stack(target_tensors)

    def _string_to_one_hot_array(self, input_string: str) -> Tuple[Tensor]:

        ind = [self.char_to_idx[ch] for ch in input_string]

        array = self._one_hot_text_data(ind)

        return array

    def _one_hot_text_data(self,
                           sequence: List):

        sequence_length = len(sequence)
        batch = torch.zeros(sequence_length, self.vocab_size)
        for i in range(sequence_length):
            batch[i, sequence[i]] = 1.0

        return Tensor(batch)

    def generate_test_data(self) -> Tuple[Tensor]:

        features_str, target_str = self.test_data[:-1], self.test_data[1:]

        X_tensors = []
        y_tensors = []

        N = len(self.test_data)

        for start in range(0, N, self.max_len):

            features_str, target_str =\
             self.test_data[start:start+self.max_len],\
             self.test_data[start+1:start+self.max_len+1]

            features_array, target_array =\
                self._string_to_one_hot_array(features_str),\
                self._string_to_one_hot_array(target_str)

            X_tensors.append(features_array)
            y_tensors.append(target_array)

        return torch.stack(X_tensors), torch.stack(y_tensors)
