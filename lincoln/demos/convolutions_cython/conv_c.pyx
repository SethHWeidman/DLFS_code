
import cython

import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def _pad_1d_obs_conv(np.ndarray obs,
                     np.int pad):
    cdef np.ndarray a = np.zeros(pad, dtype=DTYPE)
    z = np.concatenate([a, obs, a])
    return z


def _pad_1d_conv(np.ndarray inp,
                 np.int pad):
    return np.stack([_pad_1d_obs_conv(obs, pad) for obs in inp])


def _pad_2d_obs_conv(np.ndarray inp,
                     np.int pad):
    cdef np.ndarray inp_pad = _pad_1d_conv(inp, pad)
    cdef np.ndarray other = np.zeros((pad, inp.shape[0] + pad * 2))
    return np.concatenate([other, inp_pad, other])


def _pad_2d_conv(np.ndarray inp,
                 np.int pad):
    return np.stack([_pad_2d_obs_conv(obs, pad) for obs in inp])


def _compute_output_obs_conv(np.ndarray obs,
                             np.ndarray param):
    '''
    Obs is a 2d square Tensor, so is param
    '''
    cdef int param_size = param.shape[0]
    cdef int param_mid = param_size // 2

    cdef np.ndarray obs_pad = _pad_2d_obs_conv(obs, param_mid)

    cdef np.ndarray out = np.zeros_like(obs, dtype=DTYPE)

    cdef int o_w, o_h, p_w, p_h
    for o_w in range(out.shape[0]):
        for o_h in range(out.shape[1]):
            for p_w in range(param_size):
                for p_h in range(param_size):
                    out[o_w][o_h] += param[p_w][p_h] * obs_pad[o_w+p_w][o_h+p_h]
    return out


def _compute_output_obs_conv(np.ndarray obs,
                             np.ndarray param):
    '''
    Obs is a 2d square Tensor, so is param
    '''
    cdef int param_size = param.shape[0]
    cdef int param_mid = param_size // 2

    cdef np.ndarray obs_pad = _pad_2d_obs_conv(obs, param_mid)

    cdef np.ndarray out = np.zeros_like(obs, dtype=DTYPE)

    cdef int o_w, o_h, p_w, p_h
    for o_w in range(out.shape[0]):
        for o_h in range(out.shape[1]):
            for p_w in range(param_size):
                for p_h in range(param_size):
                    out[o_w][o_h] += param[p_w][p_h] * obs_pad[o_w+p_w][o_h+p_h]
    return out


def _compute_output_conv(np.ndarray inp,
                         np.ndarray param):
    return np.stack([_compute_output_obs_conv(obs, param) for obs in inp])


def _compute_grads_obs_conv(np.ndarray input_obs,
                            np.ndarray output_grad_obs,
                            np.ndarray param):
    '''
    Obs is a 2d square Tensor, so is param
    '''
    cdef int param_size = param.shape[0]
    cdef int param_mid = param_size // 2

    cdef np.ndarray output_pad_obs = _pad_2d_obs_conv(output_grad_obs, param_mid)

    cdef np.ndarray input_grad_obs = np.zeros_like(input_obs, dtype=DTYPE)

    cdef int i_w, i_h, p_w, p_h
    for i_w in range(input_obs.shape[0]):
        for i_h in range(input_obs.shape[1]):
            for p_w in range(param_size):
                for p_h in range(param_size):
                    input_grad_obs[i_w][i_h] += output_pad_obs[i_w+param_size-p_w-1][i_h+param_size-p_h-1] \
                    * param[p_w][p_h]

    return input_grad_obs


def _compute_grads_conv(np.ndarray inp,
                       np.ndarray output_grad,
                       np.ndarray param):
    return np.stack([_compute_grads_obs_conv(inp[i], output_grad[i], param) for i in range(inp.shape[0])])


def _param_grad_conv(np.ndarray inp,
                     np.ndarray output_grad,
                     np.ndarray param):
    '''
    Obs is a 2d square Tensor, so is param
    '''
    cdef int param_size = param.shape[0]
    cdef int param_mid = param_size // 2
    cdef np.ndarray inp_pad = _pad_2d_conv(inp, param_mid)
    cdef np.ndarray param_grad = np.zeros_like(param)

    cdef int img_w = output_grad.shape[1]
    cdef int img_h = output_grad.shape[2]

    cdef int i, o_w, o_h, p_w, p_h
    for i in range(inp.shape[0]):
        for o_w in range(img_w):
            for o_h in range(img_h):
                for p_w in range(param_size):
                    for p_h in range(param_size):
                        param_grad[p_w][p_h] += inp_pad[i][o_w+p_w][o_h+p_h] \
                        * output_grad[i][o_w][o_h]

    return param_grad
