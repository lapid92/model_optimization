# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
import operator

import numpy as np
import torch
from torch.nn import ConvTranspose2d, Linear

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.substitutions.batchnorm_folding import BatchNormalizationFolding
from model_compression_toolkit.core.common.substitutions.linear_mul_folding import LinearMulFolding
from model_compression_toolkit.core.pytorch.constants import KERNEL, BIAS, GAMMA, BETA, MOVING_MEAN, MOVING_VARIANCE, \
    EPSILON, USE_BIAS, GROUPS, IN_CHANNELS, OUT_CHANNELS


def linear_mul_node_matchers() -> [BaseNode, BaseNode]:
    """
    Function generates matchers for matching:
    (Linear, Dense)-> Mul.

    Returns:
        Matcher for Mul nodes, and source nodes.
    """
    mul_node = NodeOperationMatcher(torch.mul) | \
               NodeOperationMatcher(torch.multiply) | \
               NodeOperationMatcher(operator.mul)
    source_node = NodeOperationMatcher(Linear)
    return mul_node, source_node


def update_kernel_for_mul_folding_fn(linear_node: BaseNode,
                                     kernel: np.ndarray,
                                     weights_scale: np.ndarray) -> [np.ndarray, str]:
    return weights_scale[:, np.newaxis] * kernel, KERNEL


def pytorch_linear_mul_folding() -> LinearMulFolding:
    bn_node, source_node = linear_mul_node_matchers()
    return LinearMulFolding(source_node,
                            bn_node,
                            update_kernel_for_mul_folding_fn,
                            KERNEL,
                            BIAS,
                            GAMMA,
                            BETA,
                            MOVING_MEAN,
                            MOVING_VARIANCE,
                            EPSILON,
                            USE_BIAS,
                            layer_name_str=None,  # torch.nn.Modules don't have an attribute 'name'
                            )
