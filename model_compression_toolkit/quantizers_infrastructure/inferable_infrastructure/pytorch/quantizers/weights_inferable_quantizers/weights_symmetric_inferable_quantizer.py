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
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.base_inferable_quantizer import \
    mark_quantizer, \
    QuantizationTarget

if FOUND_TORCH:
    import torch
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizer_utils import \
        to_torch_tensor, \
        get_working_device
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers.base_symmetric_inferable_quantizer import \
        BaseSymmetricInferableQuantizer


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.SYMMETRIC],
                    quantizer_type=None)
    class WeightsSymmetricInferableQuantizer(BaseSymmetricInferableQuantizer):
        """
        Class for quantizing weights using a symmetric quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: np.ndarray,
                     per_channel: bool,
                     channel_axis: int = None,
                     input_rank: int = None
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: Axis of input to apply per-channel quantization on.
                input_rank: number of dimensions of input tensor the quantizer quantizes
            """

            super(WeightsSymmetricInferableQuantizer, self).__init__(threshold=threshold,
                                                                     num_bits=num_bits,
                                                                     signed=True)
            if per_channel:
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                assert len(
                    threshold) >= 1, f'In per-channel quantization threshold should be of length >= 1 but is ' \
                                     f'{len(threshold)}'
            else:
                assert len(
                    threshold) == 1, f'In per-tensor quantization threshold should be of length 1 but is {len(threshold)}'

            self.per_channel = per_channel
            self.channel_axis = channel_axis
            self.input_rank = input_rank

            if per_channel:
                assert input_rank is not None, f'Input rank is missing in per channel quantization'
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                self.reshape_vec = list(torch.ones(self.input_rank, dtype=torch.int32))
                self.reshape_vec[channel_axis] = self.scales.size
                self.scales = to_torch_tensor(self.scales.reshape(self.reshape_vec)).to(get_working_device())
                self.zero_points = torch.zeros(self.reshape_vec, dtype=torch.int32).to(get_working_device())
            else:
                self.reshape_vec = None
                self.scales = to_torch_tensor(self.scales).to(get_working_device())
                self.zero_points = torch.zeros(len(threshold), dtype=torch.int32).to(get_working_device())

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            inputs.requires_grad = False
            return (torch.clamp(torch.round(inputs / self.scales + self.zero_points),
                                self.min_quantized_domain, self.max_quantized_domain) -
                    self.zero_points) * self.scales


else:
    class WeightsSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsSymmetricInferableQuantizer. '
                            'Could not find torch package.')
