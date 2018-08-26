import torch
from torch import Tensor

def sigmoid(x: Tensor, 
            deriv: bool=False) -> Tensor:
    if deriv:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + torch.exp(-1.0 * x))
    

def to_2d(a: Tensor, 
          type: str="col") -> Tensor:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.dim() == 1, \
    "Input tensors must be 1 dimensional"
    
    if type == "col":        
        return a.reshape(tensor_size(a), 1)
    elif type == "row":
        return a.reshape(1, tensor_size(a))
    
def tensor_size(tensor: Tensor) -> int:
    '''
    Returns the number of elements in a 1D Tensor
    '''
    assert tensor.dim() == 1, \
    "Input tensors must be 1 dimensional"
    
    return list(tensor.size())[0]