# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
# ==============================================================================

import numpy as np

from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.base_inferable_quantizer import QuantizationTarget, \
    mark_quantizer

if FOUND_TORCH:
    import torch
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizer_utils import get_working_device, \
    fix_range_to_include_zero, to_torch_tensor
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers.base_uniform_inferable_quantizer import \
        BaseUniformInferableQuantizer


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.UNIFORM],
                    quantizer_type=None)
    class WeightsUniformInferableQuantizer(BaseUniformInferableQuantizer):
        """
        Class for quantizing weights using a uniform quantizer
        """

        def __init__(self,
                     num_bits: int,
                     min_range: np.ndarray,
                     max_range: np.ndarray,
                     per_channel: bool,
                     channel_axis: int = None,
                     input_rank: int = None
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min quantization range for quantizing weights
                max_range: max quantization range for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: Axis of input to apply per-channel quantization on.
                input_rank: number of dimensions of input tensor the quantizer quantizes
            """
            super(WeightsUniformInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                   min_range=min_range,
                                                                   max_range=max_range)

            # Align mix/max numpy arrays so they are torch Tensors on the working device
            min_range = to_torch_tensor(min_range).to(get_working_device())
            max_range = to_torch_tensor(max_range).to(get_working_device())

            self.per_channel = per_channel
            self.channel_axis = channel_axis
            self.input_rank = input_rank

            min_range, max_range = fix_range_to_include_zero(min_range,
                                                             max_range,
                                                             num_bits)
            # Compute the step size of quantized values.
            self.scales = (max_range - min_range) / (2 ** num_bits - 1)
            self.zero_points = -(
                        min_range / self.scales).int()  # zp has to be positive, and a <=0, so we multiply by -1

            if per_channel:
                assert input_rank is not None, f'Input rank is missing in per channel quantization'
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                self.reshape_vec = list(torch.ones(self.input_rank, dtype=torch.int32))
                self.reshape_vec[channel_axis] = self.scales.size().numel()
                self.scales = to_torch_tensor(self.scales.reshape(tuple(self.reshape_vec))).to(get_working_device())
                self.zero_points = to_torch_tensor(self.zero_points.reshape(self.reshape_vec)).to(get_working_device())
            else:
                self.reshape_vec = None
                self.scales = to_torch_tensor(self.scales).to(get_working_device())
                self.zero_points = to_torch_tensor(self.zero_points).to(get_working_device())

        def __call__(self,
                     inputs: torch.Tensor) -> torch.Tensor:
            """
            Weight fake quantizer
            Args:
                inputs: weights to quantize.

            Returns:
                quantized weights
            """
            inputs.requires_grad = False
            return (torch.clamp(torch.round(inputs / self.scales + self.zero_points),
                                self.min_quantized_domain, self.max_quantized_domain) -
                    self.zero_points) * self.scales


else:
    class WeightsUniformInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.error('Installing torch is mandatory '
                         'when using WeightsUniformInferableQuantizer. '
                         'Could not find torch package.')
