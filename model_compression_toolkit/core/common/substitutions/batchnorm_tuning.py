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


import copy

import numpy as np
from torch import nn

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.pytorch.constants import NUM_FEATURES, EPSILON, MOMENTUM, AFFINE, \
    TRACK_RUNNING_STATS, GAMMA, BETA, MOVING_MEAN, MOVING_VARIANCE, OUT_CHANNELS


class BatchNormalizationTuning(common.BaseSubstitution):
    """
    """

    def __init__(self,
                 source_node: NodeOperationMatcher):
        """
        Matches: Conv node (source node).

        Args:
            source_node: Node matcher for convolution type nodes.
        """
        super().__init__(matcher_instance=source_node)

    def substitute(self,
                   graph: Graph,
                   edge_nodes: BaseNode) -> Graph:
        """
        Reconstruct BatchNormalization after linear layers.

        Args:
            graph: Graph we apply the substitution on.
            edge_nodes: Linear node.

        Returns:
            Graph after applying the substitution.
        """
        conv_node = edge_nodes
        # Check for bn folding sign
        # check better
        if not conv_node.name.endswith('_bn'):
            return graph

        if len(graph.get_next_nodes(conv_node)) > 1:
            return graph

        out_node = graph.out_edges(conv_node)[0].sink_node
        orig_edge = graph.out_edges(conv_node)[0]

        # Think maybe reconstruct eps and momentum from the original bn
        eps = 1e-5
        momentum = 0.01

        orig_gamma = conv_node.prior_info.std_output

        beta = conv_node.prior_info.mean_output
        moving_mean = beta
        moving_var = np.power(orig_gamma, 2)
        gamma = np.sqrt(moving_var + eps)

        bn_node_weights = {GAMMA: gamma,
                           BETA: beta,
                           MOVING_MEAN: moving_mean,
                           MOVING_VARIANCE: moving_var}

        bn_node = BaseNode(name=conv_node.name + '_bn',
                           framework_attr={NUM_FEATURES: conv_node.framework_attr[OUT_CHANNELS],
                                           EPSILON: eps,
                                           MOMENTUM: momentum,
                                           AFFINE: True,
                                           TRACK_RUNNING_STATS: True},
                           input_shape=conv_node.output_shape,
                           output_shape=conv_node.output_shape,
                           weights=bn_node_weights,
                           layer_class=nn.BatchNorm2d)
        bn_node.candidates_quantization_cfg = copy.deepcopy(conv_node.candidates_quantization_cfg)

        bn_node.candidates_quantization_cfg[0].weights_quantization_cfg.enable_weights_quantization = False
        bn_node.candidates_quantization_cfg[0].activation_quantization_cfg.enable_activation_quantization = False

        graph.add_node_with_in_edges(bn_node, [conv_node])
        graph.add_edge(bn_node, out_node, **orig_edge.get_attributes())
        graph.remove_edge(conv_node, out_node)

        # Depend on the location of the substitution
        try:
            in_stats = graph.get_in_stats_collector(conv_node)
            out_stats = graph.get_out_stats_collector(conv_node)
            graph.set_out_stats_collector_to_node(bn_node, out_stats)
            graph.node_to_in_stats_collector.update({bn_node: in_stats})
        except:
            pass

        return graph
