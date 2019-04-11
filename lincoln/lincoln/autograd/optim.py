from lincoln.autograd.model import Model


class Optim:
    def __init__(self, ) -> None:
        pass

    def step(self, model: Model) -> None:
        raise NotImplementedError


class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, model: Model) -> None:
        for parameter in model.parameters():
            parameter -= parameter.grad * self.lr
