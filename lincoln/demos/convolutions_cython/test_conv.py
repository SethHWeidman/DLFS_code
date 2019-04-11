# Add Lincoln to system path
import sys
sys.path.append("/Users/seth/development/lincoln/")

import time

import numpy as np

from torch import Tensor
import torch

from lincoln.operations.base import Operation, ParamOperation
from lincoln.operations.dense import WeightMultiply
from lincoln.layers import Layer, Dense
from lincoln.operations.activations import Activation, LinearAct


class Conv2D(ParamOperation):

    def __init__(self,
                 param: Tensor):
        super().__init__(param)
        self.param_size = param.shape[0]
        self.param_pad = self.param_size // 2

    def _pad_1d_obs(self, obs: Tensor) -> Tensor:
        z = torch.Tensor([0])
        z = z.repeat(self.param_pad)
        return torch.cat([z, obs, z])

    def _pad_1d(self, inp: Tensor) -> Tensor:
        outs = [self._pad_1d_obs(obs) for obs in inp]
        return torch.stack(outs)

    def _pad_2d_obs(self,
                    inp: Tensor):

        inp_pad = self._pad_1d(inp)
        other = torch.zeros(self.param_pad, inp.shape[0] + self.param_pad * 2)
        return torch.cat([other, inp_pad, other])

    def _pad_2d(self, inp: Tensor):

        outs = [self._pad_2d_obs(obs) for obs in inp]
        return torch.stack(outs)

    def _compute_output_obs(self,
                            obs: Tensor):
        '''
        Obs is a 2d square Tensor, so is param
        '''
        obs_pad = self._pad_2d_obs(obs)

        out = torch.zeros(obs.shape)

        for o_w in range(out.shape[0]):
            for o_h in range(out.shape[1]):
                for p_w in range(self.param_size):
                    for p_h in range(self.param_size):
                        out[o_w][o_h] += self.param[p_w][p_h] * obs_pad[o_w+p_w][o_h+p_h]
        return out

    def _outputs(self):

        outs = [self._compute_output_obs(obs) for obs in self.inputs]
        return torch.stack(outs)

    def _compute_grads_obs(self,
                           input_obs: Tensor,
                           output_grad_obs: Tensor) -> Tensor:

        output_obs_pad = self._pad_2d_obs(output_grad_obs)
        input_grad = torch.zeros_like(input_obs)

        for i_w in range(input_obs.shape[0]):
            for i_h in range(input_obs.shape[1]):
                for p_w in range(self.param_size):
                    for p_h in range(self.param_size):
                        input_grad[i_w][i_h] += output_obs_pad[i_w+self.param_size-p_w-1][i_h+self.param_size-p_h-1] \
                        * self.param[p_w][p_h]

        return input_grad

    def _input_grads(self, output_grad: Tensor) -> Tensor:

        grads = [self._compute_grads_obs(self.inputs[i], output_grad[i]) for i in range(output_grad.shape[0])]

        return torch.stack(grads)

    def _param_grad(self, output_grad: Tensor) -> Tensor:

        inp_pad = self._pad_2d(self.inputs)

        param_grad = torch.zeros_like(self.param)
        img_shape = output_grad.shape[1:]

        for i in range(self.inputs.shape[0]):
            for o_w in range(img_shape[0]):
                for o_h in range(img_shape[1]):
                    for p_w in range(self.param_size):
                        for p_h in range(self.param_size):
                            param_grad[p_w][p_h] += inp_pad[i][o_w+p_w][o_h+p_h] \
                            * output_grad[i][o_w][o_h]
        return param_grad

from conv_c import (_pad_1d_obs_conv,
                    _pad_1d_conv,
                    _pad_2d_obs_conv,
                    _pad_2d_conv,
                    _compute_output_obs_conv,
                    _compute_output_conv,
                    _compute_grads_obs_conv,
                    _compute_grads_conv,
                    _param_grad_conv)

class Conv2D_cy(ParamOperation):

    def __init__(self,
                 param: Tensor):
        super().__init__(param)
        self.param_size = param.shape[0]
        self.param_pad = self.param_size // 2

    def _pad_1d_obs(self, obs: Tensor) -> Tensor:
        obs_np = obs.numpy()
        return Tensor(_pad_1d_obs_conv(obs_np, self.param_pad))

    def _pad_1d(self, inp: Tensor) -> Tensor:
        inp_np = inp.numpy()
        return Tensor(_pad_1d_conv(inp_np, self.param_pad))

    def _pad_2d_obs(self,
                    inp: Tensor):
        inp_np = inp.numpy()
        return Tensor(_pad_2d_obs_conv(inp_np, self.param_pad))

    def _pad_2d(self, inp: Tensor):
        inp_np = inp.numpy()
        return Tensor(_pad_2d_conv(inp_np, self.param_pad))

    def _compute_output_obs(self,
                            obs: Tensor):
        obs_np = obs.numpy()
        return Tensor(_compute_output_obs_conv(obs_np, self.param.numpy()))

    def _outputs(self, ):
        return Tensor(_compute_output_conv(self.inputs.numpy(),
                                           self.param.numpy()))

    def _compute_grads_obs(self,
                           input_obs: Tensor,
                           output_grad_obs: Tensor) -> Tensor:
        input_obs_np = input_obs.numpy()
        output_grad_obs = output_grad_obs.numpy()
        return Tensor(_compute_grads_obs_conv(input_obs_np, output_grad_obs, self.param))

    def _input_grads(self, output_grad: Tensor) -> Tensor:
        output_grad_np = output_grad.numpy()
        return Tensor(_compute_grads_conv(self.inputs.numpy(),
                                          output_grad_np,
                                          self.param.numpy()))

    def _param_grad(self, output_grad: Tensor) -> Tensor:

        output_grad_np = output_grad.numpy()
        return Tensor(_param_grad_conv(self.inputs.numpy(),
                                       output_grad_np,
                                       self.param.numpy()))


def main(fil: Tensor,
         imgs: Tensor):

    print("Running iteration with batch size",
        imgs.shape[0])

    def _test_conv_class(class_obj):
        start = time.time()
        out = class_obj.forward(imgs)
        out_grad = torch.empty_like(out).uniform_(-1, 1)
        class_obj.backward(out_grad)
        end = time.time()
        print("One iteration with class", class_obj.__class__.__name__,
              round(end - start, 4), "seconds")
        return None

    a = Conv2D(fil)
    b = Conv2D_cy(fil)

    _test_conv_class(a)
    _test_conv_class(b)

if __name__=="__main__":
    fil = Tensor(torch.empty(3, 3).uniform_(-1, 1))
    mnist_imgs = torch.load("data/img_batch.pt")
    main(fil, mnist_imgs)
