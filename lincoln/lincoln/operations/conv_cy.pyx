
import cython

import numpy as np
cimport numpy as np

DTYPE = np.float32

def _pad_1d_obs_cy(np.ndarray obs,
                     np.int pad):
    cdef np.ndarray a = np.zeros(pad, dtype=DTYPE)
    z = np.concatenate([a, obs, a])
    return z


def _pad_1d_batch_cy(np.ndarray inp,
                 np.int pad):
    return np.stack([_pad_1d_obs_cy(obs, pad) for obs in inp])


def _pad_2d_obs_cy(np.ndarray inp,
                     np.int pad):
    cdef np.ndarray inp_pad = _pad_1d_batch_cy(inp, pad)
    cdef np.ndarray other = np.zeros((pad, inp.shape[0] + pad * 2), dtype=DTYPE)
    return np.concatenate([other, inp_pad, other])


def _pad_2d_channel_cy(np.ndarray inp,
                       np.int pad):

    return np.stack([_pad_2d_obs_cy(channel, pad) for channel in inp])


def _pad_conv_input_cy(np.ndarray inp,
                       np.int pad):
    return np.stack([_pad_2d_channel_cy(obs, pad) for obs in inp])


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _compute_output_obs_cy(np.ndarray inp,
                           np.ndarray param):

    cdef int param_size = param.shape[2]
    cdef int in_channels = inp.shape[0]
    cdef int out_channels = param.shape[0]
    cdef int param_mid = param_size // 2
    cdef int img_size = inp.shape[1]

    cdef np.ndarray obs_pad = _pad_2d_channel_cy(inp, param_mid)

    cdef np.ndarray out = np.zeros((out_channels, img_size, img_size), dtype=DTYPE)

    cdef int c_out, c_in, o_w, o_h, p_w, p_h
    for c_out in range(out_channels):
        for c_in in range(in_channels):
            for o_w in range(img_size):
                for o_h in range(img_size):
                    for p_w in range(param_size):
                        for p_h in range(param_size):
                            out[c_out][o_w][o_h] += \
                            param[c_out][c_in][p_w][p_h] * \
                            obs_pad[c_in][o_w+p_w][o_h+p_h]
    return out


def _output_cy(np.ndarray inp,
               np.ndarray param):

    return np.stack([_compute_output_obs_cy(obs, param) for obs in inp])


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _compute_grads_obs_cy(np.ndarray input_obs,
                          np.ndarray output_grad_obs,
                          np.ndarray param):

    cdef int param_size = param.shape[2]
    cdef int in_channels = param.shape[1]
    cdef int out_channels = param.shape[0]
    cdef int param_mid = param_size // 2

    cdef np.ndarray output_pad_obs = _pad_2d_channel_cy(output_grad_obs,
                                                        param_mid)

    cdef np.ndarray input_grad_obs = np.zeros_like(input_obs, dtype=DTYPE)

    cdef int c_out, c_in, i_w, i_h, p_w, p_h
    for c_out in range(out_channels):
        for c_in in range(in_channels):
            for i_w in range(input_obs.shape[1]):
                for i_h in range(input_obs.shape[2]):
                    for p_w in range(param_size):
                        for p_h in range(param_size):
                            input_grad_obs[c_in][i_w][i_h] += \
                            output_pad_obs[c_out][i_w+param_size-p_w-1][i_h+param_size-p_h-1] \
                            * param[c_out][c_in][p_w][p_h]

    return input_grad_obs


def _input_grad_cy(np.ndarray inp,
                   np.ndarray output_grad,
                   np.ndarray param):
    return np.stack([_compute_grads_obs_cy(inp[i], output_grad[i], param) for i in range(inp.shape[0])])


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _param_grad_cy(np.ndarray inp,
                   np.ndarray output_grad,
                   np.ndarray param):
    '''
    Obs is a 2d square Tensor, so is param
    '''
    cdef int param_size = param.shape[2]
    cdef int param_mid = param_size // 2
    cdef np.ndarray inp_pad = _pad_conv_input_cy(inp, param_mid)
    cdef np.ndarray param_grad = np.zeros_like(param, dtype=DTYPE)

    cdef int batch_size = inp.shape[0]
    cdef int in_channels = param.shape[1]
    cdef int out_channels = param.shape[0]
    cdef int img_w = output_grad.shape[2]
    cdef int img_h = output_grad.shape[3]

    cdef int i, c_in, c_out, o_w, o_h, p_w, p_h
    for i in range(batch_size):
        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for o_w in range(img_w):
                    for o_h in range(img_h):
                        for p_w in range(param_size):
                            for p_h in range(param_size):
                                param_grad[c_out][c_in][p_w][p_h] += \
                                inp_pad[i][c_in][o_w+p_w][o_h+p_h] \
                                * output_grad[i][c_out][o_w][o_h]
    return param_grad
