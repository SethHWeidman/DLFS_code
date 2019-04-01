import numpy as np


class Optimizer(object):
    def __init__(self,
                 lr: float = 0.01,
                 exp_decay: float = 0,
                 final_lr_linear: float = 0):
        self.lr = lr
        self.exp_decay = exp_decay # check this
        self.final_lr_linear = final_lr_linear
        if self.exp_decay or self.final_lr_linear:
            self.decay_lr = True
        else:
            self.decay_lr = False
        self.first = True

    def _decay_lr(self,
                  epoch: int = 0,
                  max_epochs: int = 0) -> None:

        self.lr_orig = self.lr

        if self.exp_decay:
            self.lr *= self.exp_decay

        elif self.final_lr_linear:
            self.lr = self.lr_orig - (self.lr_orig - self.final_lr_linear) * \
            ((epoch+1) / max_epochs)
        else:
            pass


    def step(self) -> None:

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
            self._update_rule(param=param,
                              grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 exp_decay: float = 0,
                 final_lr_linear: float = 0) -> None:
        super().__init__(lr, exp_decay, final_lr_linear)

    def _update_rule(self, **kwargs) -> None:

        update = self.lr*kwargs['grad']
        kwargs['param'] -= update



class SGDMomentum(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 exp_decay: float = 0,
                 final_lr_linear: float = 0,
                 momentum: float = 0.9) -> None:
        super().__init__(lr, exp_decay, final_lr_linear)
        self.momentum = momentum

    def step(self) -> None:
        if self.first:
            self.velocities = [np.zeros_like(param)
                               for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(self.net.params(),
                                                 self.net.param_grads(),
                                                 self.velocities):
            self._update_rule(param=param,
                              grad=param_grad,
                              velocity=velocity)

    def _update_rule(self, **kwargs) -> None:

            # Update velocity
            kwargs['velocity'] *= self.momentum
            kwargs['velocity'] += self.lr * kwargs['grad']

            # Use this to update parameters
            kwargs['param'] -= kwargs['velocity']


class AdaGrad(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 exp_decay: float = 0,
                 final_lr_linear: float = 0) -> None:
        super().__init__(lr, exp_decay, final_lr_linear)
        self.eps = 1e-7

    def step(self) -> None:
        if self.first:
            self.sum_squares = [np.zeros_like(param)
                                for param in self.net.params()]
            self.first = False

        for (param, param_grad, sum_square) in zip(self.net.params(),
                                                   self.net.param_grads(),
                                                   self.sum_squares):
            self._update_rule(param=param,
                              grad=param_grad,
                              sum_square=sum_square)

    def _update_rule(self, **kwargs) -> None:

            # Update running sum of squares
            kwargs['sum_square'] += (self.eps +
                                     np.power(kwargs['grad'], 2))

            # Scale learning rate by running sum of squareds=5
            lr = np.divide(self.lr, np.sqrt(kwargs['sum_square']))

            # Use this to update parameters
            kwargs['param'] -= lr * kwargs['grad']


class RegularizedSGD(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 alpha: float = 0.1) -> None:
        super().__init__()
        self.lr = lr
        self.alpha = alpha

    def step(self) -> None:

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):

            self._update_rule(param=param,
                              grad=param_grad)

    def _update_rule(self, **kwargs) -> None:

            # Use this to update parameters
            kwargs['param'] -= self.lr * (
                kwargs['grad'] + self.alpha * kwargs['param'])
