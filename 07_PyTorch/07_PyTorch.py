# imports
from typing import Tuple, List
from collections import deque

import torch
import torch.optim as optim
from torch.optim import Optimizer

import numpy as np
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from lincoln.utils import permute_data, assert_dim

from lincoln.pytorch.model import PyTorchModel
from lincoln.pytorch.train import PyTorchTrainer
from lincoln.pytorch.preprocessor import ConvNetPreprocessor

torch.manual_seed(20190325);


from sklearn.datasets import load_boston

boston = load_boston()

data = boston.data
target = boston.target
features = boston.feature_names

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

X_train, X_test, y_train, y_test = Tensor(X_train), Tensor(X_test), Tensor(y_train), Tensor(y_test)


class PyTorchLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

class DenseLayer(PyTorchLayer):
    def __init__(self,
                 input_size: int,
                 neurons: int,
                 activation: nn.Module = None) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, neurons)
        self.activation = activation


    def forward(self, x: Tensor) -> Tensor:

        x = self.linear(x) # does weight multiplication + bias
        if self.activation:
            x = self.activation(x)

        return x


class BostonModel(PyTorchModel):

    def __init__(self,
                 hidden_size: int = 13):
        super().__init__()
        self.dense1 = DenseLayer(13, hidden_size, activation=nn.Tanh())
        self.dense2 = DenseLayer(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:

        assert_dim(x, 2)

        assert x.shape[1] == 13

        x = self.dense1(x)
        return self.dense2(x)


# model, optimizer, loss
pytorch_boston_model = BostonModel(hidden_size=13)
optimizer = optim.SGD(pytorch_boston_model.parameters(), lr=0.001)
criterion = nn.MSELoss()


trainer = PyTorchTrainer(pytorch_boston_model, optimizer, criterion)

trainer.fit(X_train, y_train, X_test, y_test,
            epochs=300,
            eval_every=10)


torch.mean(torch.pow(pytorch_boston_model(X_test) - y_test, 2)).item()

test_pred = pytorch_boston_model(X_test).view(-1)
test_actual = y_test

test_pred = test_pred.detach().numpy()
test_actual = test_actual.detach().numpy()

import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(test_pred, test_actual)


from torchvision.datasets import MNIST
mnist_trainset = MNIST(root="../exploratory/data/", train=True, download=True, transform=None)
mnist_testset = MNIST(root="../exploratory/data/", train=False, download=True, transform=None)

mnist_train = mnist_trainset.train_data.type(torch.float32).unsqueeze(3) / 255.0
mnist_test = mnist_testset.test_data.type(torch.float32).unsqueeze(3) / 255.0


class ConvLayer(PyTorchLayer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 filter_size: int,
                 activation: nn.Module = None,
                 flatten: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, filter_size,
                              padding=filter_size // 2)
        self.activation = activation
        self.flatten = flatten

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.flatten:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        return x


class MNIST_ConvNet(PyTorchModel):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(1, 16, 5, activation=nn.Tanh())
        self.conv2 = ConvLayer(16, 8, 5, activation=nn.Tanh(), flatten=True)
        self.dense1 = DenseLayer(28 * 28 * 8, 32, activation=nn.Tanh())
        self.dense2 = DenseLayer(32, 10)

    def forward(self, x: Tensor) -> Tensor:
        assert_dim(x, 4)

        # num channgels
        try:
            assert x.shape[1] == 1
        except:
            import pdb; pdb.set_trace()

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.dense1(x)
        x = self.dense2(x)
        return x

model = MNIST_ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
preprocessor = ConvNetPreprocessor()

trainer = PyTorchTrainer(model, optimizer, criterion, preprocessor)

trainer.fit(mnist_train, mnist_trainset.train_labels,
            mnist_test, mnist_testset.test_labels,
            epochs=5,
            eval_every=1)

out = model(X_test)

(torch.max(out, dim=1)[1] == mnist_testset.test_labels).sum()

class LSTMLayer(PyTorchLayer):
    def __init__(self,
                 sequence_length: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.h_init = torch.zeros((1, hidden_size))
        self.c_init = torch.zeros((1, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = DenseLayer(hidden_size, output_size)


    def _transform_hidden_batch(self, hidden: Tensor,
                                batch_size: int,
                                before_layer: bool) -> Tensor:

        if before_layer:
            return (hidden
                    .repeat(batch_size, 1)
                    .view(batch_size, 1, self.hidden_size)
                    .permute(1,0,2))
        else:
            return (hidden
                    .permute(1,0,2)
                    .mean(dim=0))


    def forward(self, x: Tensor) -> Tensor:

        batch_size = x.shape[0]

        h_layer = self._transform_hidden_batch(self.h_init, batch_size, before_layer=True)
        c_layer = self._transform_hidden_batch(self.c_init, batch_size, before_layer=True)

        x, (h_out, c_out) = self.lstm(x, (h_layer, c_layer))

        self.h_init, self.c_init = (
            self._transform_hidden_batch(h_out, batch_size, before_layer=False).detach(),
            self._transform_hidden_batch(c_out, batch_size, before_layer=False).detach()
        )

        x = self.fc(x)

        return x


lay = LSTMLayer(sequence_length=25,
          input_size=62,
          hidden_size=100,
          output_size=128)

x = torch.randn(32, 25, 62)

lay(x).shape

class NextCharacterModel(PyTorchModel):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 256,
                 sequence_length: int = 25):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        # Note that both the output size and input size are `vocab_size`

        self.lstm = LSTMLayer(sequence_length, vocab_size, hidden_size, vocab_size)

#         self.softmax = nn.Softmax(dim = 2)


    def forward(self,
                inputs: Tensor):
        assert_dim(inputs, 3) # batch_size, sequence_length, vocab_size

        out = self.lstm(inputs)

        return out


class TextGenerationPreprocessor(PyTorchPreprocessor):
    def __init__(self):
        pass

    def transform(self, x: Tensor) -> Tensor:
        return x.permute(0, 3, 1, 2)

class LSTMTrainer(PyTorchTrainer):
    def __init__(self,
                 model: NextCharacterModel,
                 optim: Optimizer,
                 criterion: _Loss):
        super().__init__(model, optim, criterion)
        self.vocab_size = self.model.vocab_size
        self.max_len = self.model.sequence_length

    def fit(self,
            data: str,
            epochs: int=10,
            eval_every: int=1,
            batch_size: int=32,
            seed: int = 121718)-> None:

        self.data = data
        self.train_data, self.test_data = self._train_test_split_text()
        self.chars = list(set(self.data))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        torch.manual_seed(seed)

        losses = deque(maxlen=50)

        for e in range(epochs):

            batch_generator = self.generate_batches_next_char(batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.optim.zero_grad()
                outputs = self.model(X_batch)

                # reshape outputs to work with CrossEntropyLoss
                outputs = outputs.permute(0, 2, 1)

                loss = self.loss(outputs, y_batch)
                losses.append(loss.item())

                print(loss.item())
                loss.backward()
#                 for p in self.model.parameters():
#                     import pdb; pdb.set_trace()

                self.optim.step()

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
            target_indices = []

            for char in range(batch_size):

                features_str, target_str =\
                 self.train_data[ii+char:ii+char+self.max_len],\
                 self.train_data[ii+char+1:ii+char+self.max_len+1]

                features_array = self._string_to_one_hot_array(features_str)
                target_indices_seq = [self.char_to_idx[char] for char in target_str]

                features_tensors.append(features_array)
                target_indices.append(target_indices_seq)
#             import pdb; pdb.set_trace()
            yield torch.stack(features_tensors), torch.LongTensor(target_indices)

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


data = open('data/input.txt', 'r').read()
vocab_size = len(set(data))
model = NextCharacterModel(vocab_size, hidden_size=vocab_size, sequence_length=50)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                             weight_decay=1e-5)

lstm_trainer = LSTMTrainer(model, optimizer, criterion)

lstm_trainer.fit(data)
