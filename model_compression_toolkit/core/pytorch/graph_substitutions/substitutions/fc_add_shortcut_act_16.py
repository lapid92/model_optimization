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
from typing import List, Tuple
from mct_quantizers.common import constants as C
import torch
from torch.nn import LayerNorm, Linear

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, WalkMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.constants import THRESHOLD


class FCAddShortCutBits(common.BaseSubstitution):
    def __init__(self):
        linear_node = NodeOperationMatcher(Linear)
        add_node = NodeOperationMatcher(torch.add) | \
            NodeOperationMatcher(operator.add)
        super().__init__(matcher_instance=WalkMatcher([linear_node, add_node]))

    def substitute(self,
                   graph: Graph,
                   nodes: Tuple[BaseNode, BaseNode]) -> Graph:
        linear_node = nodes[0]
        add_node = nodes[1]
        add_prev_nodes = graph.get_prev_nodes(add_node)

        if linear_node not in add_prev_nodes or len(add_prev_nodes) != 2:
            return graph

        for prev_node in add_prev_nodes:
            if prev_node.type in [operator.add, torch.add]:
                output_prev_add = graph.get_next_nodes(prev_node)
                if len(output_prev_add) == 2:
                    output_prev_add_type = [next_node.type for next_node in output_prev_add]
                    if LayerNorm in output_prev_add_type and (operator.add in output_prev_add_type or torch.add in output_prev_add_type):
                        for nqc in linear_node.candidates_quantization_cfg:
                            nqc.activation_quantization_cfg.set_activation_quantization_param({C.ACTIVATION_N_BITS: 16})
                            nqc.activation_quantization_cfg.activation_n_bits = 16
        return graph




