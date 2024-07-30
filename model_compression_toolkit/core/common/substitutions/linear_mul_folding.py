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


import copy
import numpy as np
from typing import Tuple, Callable

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import EdgeMatcher, NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode


class LinearMulFolding(common.BaseSubstitution):
    """
    Fold Mul into preceding linear layers.
    """

    def __init__(self,
                 source_node: NodeOperationMatcher,
                 mul_node: NodeOperationMatcher,
                 update_kernel_for_mul_folding_fn: Callable,
                 kernel_str: str,
                 bias_str: str,
                 gamma_str: str,
                 beta_str: str,
                 moving_mean_str: str,
                 moving__variance_str: str,
                 epsilon_str: str,
                 use_bias: str,
                 layer_name_str: str):
        super().__init__(matcher_instance=EdgeMatcher(source_node, mul_node))
        self.update_kernel_for_bn_folding_fn = update_kernel_for_mul_folding_fn
        self.kernel_str = kernel_str
        self.bias_str = bias_str
        self.gamma_str = gamma_str
        self.beta_str = beta_str
        self.moving_mean_str = moving_mean_str
        self.moving__variance_str = moving__variance_str
        self.epsilon_str = epsilon_str
        self.use_bias = use_bias
        self.layer_name_str = layer_name_str

    def substitute(self,
                   graph: Graph,
                   edge_nodes: Tuple[BaseNode, BaseNode]) -> Graph:

        num_nodes_before_substition = len(graph.nodes)
        num_edges_before_substition = len(graph.edges)

        conv_node = edge_nodes[0]

        # If the linear operator is part of a reused group (it is the "base" node, or a reused node),
        # we should skip the substitution.
        if conv_node.is_reused():
            return graph

        bn_node = edge_nodes[1]

        if len(graph.get_next_nodes(conv_node)) > 1 or len(graph.get_prev_nodes(bn_node)) > 1:
            return graph

        kernel = conv_node.get_weights_by_keys(self.kernel_str)
        bias = conv_node.get_weights_by_keys(self.bias_str)
        gamma = bn_node.get_weights_by_keys(1)

        if bias is None:
            bias = 0.0
        bias = bias * gamma

        kernel, kernel_name = self.update_kernel_for_bn_folding_fn(conv_node, kernel, gamma)

        framework_attr = copy.copy(conv_node.framework_attr)
        framework_attr[self.use_bias] = True
        if self.layer_name_str is not None:
            framework_attr[self.layer_name_str] = conv_node.name + '_bn'

        weights_dict = {kernel_name: kernel,
                        self.bias_str: bias}

        conv_bn = copy.deepcopy(conv_node)
        conv_bn_name = conv_node.name + '_bn'
        conv_bn.name = conv_bn_name
        conv_bn.framework_attr = framework_attr
        conv_bn.weights = weights_dict

        graph.add_node(conv_bn)
        graph.reconnect_out_edges(current_node=bn_node, new_node=conv_bn)
        graph.reconnect_in_edges(current_node=conv_node, new_node=conv_bn)

        graph.replace_output_node(current_node=bn_node, new_node=conv_bn)

        conv_bn.prior_info = bn_node.prior_info

        graph.remove_edge(conv_node, bn_node)
        graph.remove_node(bn_node)
        graph.remove_node(conv_node)

        assert num_nodes_before_substition - len(graph.nodes) == 1
        assert num_edges_before_substition - len(graph.edges) == 1
        return graph
