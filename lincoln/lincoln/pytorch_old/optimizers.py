import torch


class Optimizer(object):
    def __init__(self):
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
                 lr: float = 0.003) -> None:
        super().__init__()
        self.lr = lr

    def _update_rule(self, **kwargs) -> None:
        # import pdb; pdb.set_trace()
        kwargs['param'].data.sub_(self.lr*kwargs['grad'].data)


class SGDMomentum(Optimizer):
    def __init__(self,
                 lr: float = 0.003,
                 momentum: float = 0.9) -> None:
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.first = True

    def step(self) -> None:
        if self.first:
            self.velocities = [torch.zeros_like(param)
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
            kwargs['velocity'].mul_(self.momentum).add_(self.lr * kwargs['grad'])

            # Use this to update parameters
            kwargs['param'].sub_(kwargs['velocity'])
