# Copyright (C) 2024 Yannick Emonds
# Modifications copyright (C) 2024 Franz Kevin Stehle Computing Systems Group, Institute of Computer Engineering, Heidelberg University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Implementation of the fliptensors method developed by Emonds et al. in "Implications of 
# Noise in Resistive Memory on Deep Neural Networks for Image Classification" (https://arxiv.org/abs/2401.05820).

from functools import lru_cache

import numpy as np
import torch

def clip_values_to_range(value: torch.Tensor, low, high, nan):
    if value is None:
        return None
    value[value < low] = low
    value[value > high] = high
    torch.nan_to_num(value, nan=nan, out=value)


# TODO: These globals could be in some helper class/in class BitFlipNoise to reduce namespace pollution.
type_widths = {torch.float16: 16, torch.half: 16, torch.bfloat16: 16, torch.int16: 16, torch.short: 16,
               torch.float32: 32, torch.float: 32, torch.complex32: 32, torch.int32: 32, torch.int: 32,
               torch.float64: 64, torch.double: 64, torch.complex64: 64, torch.cfloat: 64, torch.int64: 64, torch.long: 64,
               torch.complex128: 128, torch.cdouble: 128,
               torch.bool: 1}
scale_tensor8 = torch.tensor([np.int8(2**(7-i)) for i in range(8)], dtype=torch.int8).detach()
scale_tensor16 = torch.tensor([np.int16(2**(15-i)) for i in range(16)], dtype=torch.int16).detach()
scale_tensor32 = torch.tensor([np.int32(2**(31-i)) for i in range(32)], dtype=torch.int32).detach()
scale_tensor64 = torch.tensor([np.int64(2**(63-i)-1) for i in range(64)], dtype=torch.int64).add_(1).detach()
# scale_tensor128 = torch.tensor([2**(127-i)] for i in range(128))


def get_bitwidth(var):
    """The same behavior can be achieved using the PyTorch functions torch.finfo and torch.iinfo."""
    return type_widths[var.dtype]

@lru_cache(4)
def _sample_bitflips_impl(b, l, w, bw, rng, device: str):

    flip_shape = [b, l, w, bw]

    return rng.sample(flip_shape).view(-1, *flip_shape).int().detach().cuda()


def sample_bitflips(parameter, rng, device: str):
    flip_shape = list(parameter.size())
    flip_shape.append(get_bitwidth(parameter))

    return _sample_bitflips_impl(*flip_shape, rng, device)


def reduce_fliptensor(bitwidth: int, fliptensor: torch.Tensor, device: str):
    """Reduce fliptensor from tensor of single bits to integer according to bitwidth."""
    scaler = scale_tensor32
    if bitwidth == 64:
        scaler = scale_tensor64
        fliptensor = fliptensor.long()
    elif bitwidth == 16:
        scaler = scale_tensor16
        fliptensor = fliptensor.short()
    elif bitwidth == 8:
        scaler = scale_tensor8
        fliptensor = torch.tensor(fliptensor, dtype=torch.int8, requires_grad=False)
    elif bitwidth == 128:
        raise NotImplementedError("No 128-bit wide integer type available.")

    fliptensor = fliptensor.detach()

    scaler = scaler.cuda().detach()
    # torch.inner cannot be used with integer on CUDA backend (only floats in CUBLAS)
    # return torch.inner(fliptensor, scaler)
    return torch.sum(fliptensor*scaler, dim=-1).detach()


def apply_bitflips(par, fliptensor):
    bitwidth = get_bitwidth(par)

    if bitwidth == 32:
        par_as_int = par.view(torch.int32)
    elif bitwidth == 64:
        par_as_int = par.view(torch.int64)
    elif bitwidth == 16:
        par_as_int = par.view(torch.int16)
    elif bitwidth == 8:
        par_as_int = par.view(torch.int8)
    else:
        raise TypeError("Unknown bitwidth.")

    par_as_int = par_as_int.detach()

    # (p | f0) ^ (p & f1)

    par_as_int_flipped = torch.empty(par.shape,
                                        dtype=par_as_int.dtype,
                                        device=par.device).detach()

    torch.bitwise_xor(torch.bitwise_or(par_as_int, fliptensor[0]),
                            torch.bitwise_and(par_as_int, fliptensor[1]),
                            out=par_as_int_flipped)

    par_flipped = par_as_int_flipped.view(dtype=par.dtype)

     # This is completely arbitrary.
    par_flipped = torch.nan_to_num(par_flipped, nan=0.0)

    del par_as_int
    del par_as_int_flipped

    return par_flipped


def add_noise(value: torch.Tensor, rng, device: str = 'cpu'):
    """Draw bit-flip tensors and apply it to value (in-place)."""
    if value is None:
        return
    # rng = torch.distributions.bernoulli.Bernoulli(torch.Tensor(probabilities))
    flip_tensor = sample_bitflips(value, rng, device)
    flip_tensor = reduce_fliptensor(get_bitwidth(value), flip_tensor, device)
    value.data = apply_bitflips(value.data, flip_tensor)


def get_device_str(t: torch.Tensor) -> str:
    device = 'cuda'
    try:
        if t.get_device() < 0:
            device = 'cpu'
    except RuntimeError:
        device = 'cpu'
    finally:
        return device


class BitFlipNoise:
    """Applies bit-flips to input value according to given probabilities.

    Two bit-vectors are drawn from a Bernoulli distribution representing the flip from 0 to 1 and vice versa.
    The length of each vector coincides with the bit-width of the input value.
    To apply the bit-flips, the following bitwise operation is performed:

    .. math::
        (v \mid f_{1}) \wedge (v \:\& f_{2})

    where :math:`v` is the input value, :math:`f_{1}` is the flip vector from 0 to 1 and :math:`f_{2}` is the flip vector
    from 1 to 0.

    Params:
        - probabilities: tuple of the bit-flip probability from 0 to 1 and from 1 to 0 (in this order).
    """
    def __init__(self, probabilities: tuple):
        self.full_rng = torch.distributions.bernoulli.Bernoulli(torch.tensor(probabilities,
                                                                                dtype=torch.float32))

    def add_noise(self, value: torch.Tensor):

        with torch.no_grad():
            """Draw bit-flip tensors and apply it to value (in-place)."""

            flip_tensor = sample_bitflips(value,
                                            self.full_rng,
                                            device='cuda')

            flip_tensor = reduce_fliptensor(get_bitwidth(value),
                                                        flip_tensor,
                                                        device='cuda')
            value.data = apply_bitflips(value.data,
                                            flip_tensor)

    def __call__(self, value: torch.Tensor):
        self.add_noise(value)
