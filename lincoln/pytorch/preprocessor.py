from torch import Tensor


class PyTorchPreprocessor():
    def __init__(self):
        pass

    def transform(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class ConvNetPreprocessor(PyTorchPreprocessor):
    def __init__(self):
        pass

    def transform(self, x: Tensor) -> Tensor:
        return x.permute(0, 3, 1, 2)
