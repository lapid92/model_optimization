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
from typing import List
from mct_quantizers.common import constants as C
import torch
from torch.nn import LayerNorm

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.constants import THRESHOLD


class AddShortCutBits(common.BaseSubstitution):
    def __init__(self):
        add_node = NodeOperationMatcher(torch.add) | \
            NodeOperationMatcher(operator.add)
        super().__init__(matcher_instance=add_node)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        next_nodes = graph.get_next_nodes(node)
        if len(next_nodes) == 2:

            next_nodes_type = [next_node.type for next_node in next_nodes]

            if LayerNorm in next_nodes_type and (operator.add in next_nodes_type or torch.add in next_nodes_type):
                for nqc in node.candidates_quantization_cfg:
                    # nqc.activation_quantization_cfg.set_activation_quantization_param({C.ACTIVATION_N_BITS: 16})
                    nqc.activation_quantization_cfg.activation_n_bits = 16
        if len(next_nodes) == 1:
            if next_nodes[0].type == LayerNorm:
                for nqc in node.candidates_quantization_cfg:
                    # nqc.activation_quantization_cfg.set_activation_quantization_param({C.ACTIVATION_N_BITS: 16})
                    nqc.activation_quantization_cfg.activation_n_bits = 16
        return graph
        # if len(next_nodes) != 2:
        #     return graph
        #
        # next_nodes_type = [next_node.type for next_node in next_nodes]
        #
        # if LayerNorm in next_nodes_type and (operator.add in next_nodes_type or torch.add in next_nodes_type):
        #     for nqc in node.candidates_quantization_cfg:
        #         nqc.activation_quantization_cfg.set_activation_quantization_param({C.ACTIVATION_N_BITS: 16})
        #         nqc.activation_quantization_cfg.activation_n_bits = 16
        # return graph




