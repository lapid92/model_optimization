# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import numpy as np

from torch.nn import LayerNorm

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode


class LayerNormSubs(common.BaseSubstitution):
    def __init__(self):
        ln_node = NodeOperationMatcher(LayerNorm)
        super().__init__(matcher_instance=ln_node)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:

        first_node = graph.get_prev_nodes(node)[0]
        if first_node.type != operator.add:
            return graph

        layer_norm_node_name = node.name
        sub_node_name = layer_norm_node_name + '_prev_sub'

        sub_node = common.graph.functional_node.FunctionalNode(sub_node_name,
                                                               {},
                                                               node.input_shape,
                                                               node.input_shape,
                                                               weights={},
                                                               quantization_attr={},
                                                               functional_op=operator.sub,
                                                               op_call_kwargs={},
                                                               op_call_args=[float(0)],
                                                               layer_class=operator.sub)
        # Create prior info for the add node
        if node.prior_info is not None:
            sub_node.prior_info = node.prior_info

        graph.add_node(sub_node)
        e_attr = graph.get_edge_data(first_node, node)
        e_attr = list(e_attr.values())[0]
        graph.add_edge(first_node, sub_node, **e_attr)
        graph.add_edge(sub_node, node, **e_attr)
        graph.remove_edge(first_node, node)
        return graph
