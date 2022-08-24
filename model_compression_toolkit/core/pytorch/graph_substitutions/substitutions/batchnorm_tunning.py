# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from torch.nn import Conv2d, ConvTranspose2d

from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.substitutions.batchnorm_tuning import BatchNormalizationTuning


def batchnorm_tuning_node_matchers():
    """
    Function generates matchers for matching:
    (Conv2d, ConvTranspose2d).

    Returns:
        Matcher for conv nodes.
    """
    source_node = NodeOperationMatcher(Conv2d) | \
                  NodeOperationMatcher(ConvTranspose2d)
    return source_node


def pytorch_batchnorm_tuning() -> BatchNormalizationTuning:
    """

    Returns:
        A BatchNormalizationTuning initialized for Pytorch models.
    """
    source_node = batchnorm_tuning_node_matchers()
    return BatchNormalizationTuning(source_node)
