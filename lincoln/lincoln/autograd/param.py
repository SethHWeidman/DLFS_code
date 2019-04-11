import numpy as np
from lincoln.autograd.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        data = np.random.randn(*shape)
        super().__init__(data)
