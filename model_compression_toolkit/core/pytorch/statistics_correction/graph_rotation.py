# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import operator

import numpy as np
import torch
from torch import nn

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.edge import EDGE_SOURCE_INDEX, EDGE_SINK_INDEX
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.constants import IN_CHANNELS, OUT_CHANNELS, BIAS, KERNEL_SIZE
from model_compression_toolkit.core.pytorch.statistics_correction.hadamard_utils import random_hadamard_matrix
from model_compression_toolkit.logger import Logger

NUM_BRANCHES = 2


def is_shortcut_node(node):
    # TODO:
    # add cat with terms
    return isinstance(node, FunctionalNode) and node.type in [torch.add, operator.add]


def is_norm(n):
    return n.type in [torch.nn.LayerNorm, torch.nn.RMSNorm, torch.nn.Identity]


def is_linear(n):
    return n.type in [torch.nn.Linear]


def is_dropout(n):
    return n.type in [torch.nn.Dropout, torch.nn.functional.dropout]


# TODO:
# check for shape chn is same
def is_reshape(n):
    return n.type in [torch.reshape, torch.Tensor.view, torch.permute, torch.Tensor.contiguous, torch.roll]


def is_transparent(n):
    return is_dropout(n) or is_reshape(n)


def get_next_non_transparent_nodes(graph, node):
    result = set()
    visited = set()
    frontier = [node]

    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)

        for next_node in graph.get_next_nodes(current):
            if is_transparent(next_node):
                frontier.append(next_node)
            else:
                result.add(next_node)

    return list(result)


def get_prev_non_transparent_nodes(graph, node):
    result = set()
    visited = set()
    frontier = [node]

    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)

        for prev_node in graph.get_prev_nodes(current):
            if is_transparent(prev_node):
                frontier.append(prev_node)
            else:
                result.add(prev_node)

    return list(result)


def get_single_shortcut_node_successor(graph, n):
    """Follow transparent nodes to find a single downstream shortcut node"""
    visited = set()
    frontier = list(graph.get_next_nodes(n))
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        if is_shortcut_node(current):
            return current
        elif is_transparent(current):
            frontier.extend(graph.get_next_nodes(current))
    return None


def user_count(graph, n):
    users = []
    next_nodes = graph.get_next_nodes(n)
    for user in next_nodes:
        if is_transparent(user):
            for trans_user in graph.get_next_nodes(user):
                users.append(trans_user)
        else:
            users.append(user)
    valid_users = [u for u in users if is_shortcut_node(u)]
    # last
    if len(valid_users) == 0 and len(users) == 1:
        return 1
    elif len(valid_users) > 0:
        return len(users)
    else:
        return 0


def get_rotationable_chains(graph):
    chains = []
    visited = set()

    nodes = list(graph.get_topo_sorted_nodes())
    for idx, node in enumerate(nodes):
        if node in visited or is_transparent(node):
            continue

        # Check if node has NUM_BRANCHES users, and one is shortcut_node
        users = graph.get_next_nodes(node)
        if len(users) > 0:
            users = [
                node
                for user in users
                for node in (
                    get_next_non_transparent_nodes(graph, user)
                    if is_transparent(user) else [user]
                )
            ]
        else:
            continue
        if user_count(graph, node) != NUM_BRANCHES:
            continue

        possible_shortcut_nodes = [u for u in users if is_shortcut_node(u)]
        if len(possible_shortcut_nodes) != 1:
            continue

        # Start a potential chain
        chain = [node]
        visited.add(node)

        current = possible_shortcut_nodes[0]
        chain.append(current)
        visited.add(current)
        while True:
            node_outputs = get_single_shortcut_node_successor(graph, current)

            if node_outputs is None or node_outputs in visited:
                break

            chain.append(node_outputs)
            visited.add(node_outputs)
            current = node_outputs

        # TODO:
        # Why 2?
        if len(chain) < 2:
            continue

        first, *middle, last = chain

        valid = (
                len(graph.get_prev_nodes(first)) == 1 and user_count(graph, first) == 2 and
                all(len(graph.get_prev_nodes(n)) == 2 and user_count(graph, n) == 2 for n in middle) and
                len(graph.get_prev_nodes(last)) == 2 and user_count(graph, last) == 1
        )

        if valid:
            chains.append(chain)
    for i, chain in enumerate(chains):
        print(f"\nChain {i + 1}:")
        for node in chain:
            print(f"  {node.name}: inputs={len(graph.get_prev_nodes(node))}, outputs={user_count(graph, node)}")
    return chains


def check_block(graph, first_node, second_node):
    # Between 2 residuals we assume the block (or maybe sub-block) will look like:
    # first_add
    # norm (LN/RMSNorm/Identity) + Linear
    # ...
    # Linear + (maybe mul?) + maybe dropout
    # second_add
    linear_nodes = []
    norm_nodes = []
    next_first_node = graph.get_next_nodes(first_node)
    next_first_node = [
        node
        for user in next_first_node
        for node in (
            get_next_non_transparent_nodes(graph, user)
            if is_transparent(user) else [user]
        )
    ]
    next_first_node_wo_shortcut_node = [nfn for nfn in next_first_node if not is_shortcut_node(nfn)]

    # TODO:
    # Assume only one node?
    # Need to fix for more branches
    if is_norm(next_first_node_wo_shortcut_node[0]):
        norm_nodes.append(next_first_node_wo_shortcut_node[0])
        next_first_node_wo_shortcut_node = graph.get_next_nodes(next_first_node_wo_shortcut_node[0])
    next_first_node_wo_shortcut_node = [
        node
        for user in next_first_node_wo_shortcut_node
        for node in (
            get_next_non_transparent_nodes(graph, user)
            if is_transparent(user) else [user]
        )
    ]

    # TODO:
    # Assume only one node?
    # Need to fix for more branches
    if is_linear(next_first_node_wo_shortcut_node[0]):
        linear_nodes.append(next_first_node_wo_shortcut_node[0])

    prev_second_node = graph.get_prev_nodes(second_node)
    prev_second_node_wo_shortcut_node = [
        node
        for user in prev_second_node
        for node in (
            get_prev_non_transparent_nodes(graph, user)
            if is_transparent(user) else [user]
        )
    ]
    if first_node in prev_second_node_wo_shortcut_node:
        prev_second_node_wo_shortcut_node.remove(first_node)

    # TODO:
    # Assume only one node?
    # Need to fix for more branches
    if is_linear(prev_second_node_wo_shortcut_node[0]):
        linear_nodes.append(prev_second_node_wo_shortcut_node[0])
    return linear_nodes, norm_nodes


def random_orthogonal_matrix(size):
    random_matrix = np.random.randn(size, size).astype(np.float32)
    q, r = np.linalg.qr(random_matrix)
    q *= np.sign(np.diag(r))[np.newaxis, :].astype(np.float32)
    return q


def get_orthogonal_matrix(size, mode="hadamard"):
    if mode == "random":
        return random_orthogonal_matrix(size)
    elif mode == "hadamard":
        return random_hadamard_matrix(size)
    return False


def rotate_output_linear(node, R):
    W = node.weights['weight']
    node.set_weights_by_keys('weight', np.matmul(R, W))
    if 'bias' in node.weights.keys() and node.weights['bias'] is not None:
        node.set_weights_by_keys('bias', np.matmul(R, node.weights['bias']))

    return node


def rotate_input_linear(node, R):
    W = node.weights['weight']
    node.set_weights_by_keys('weight', np.matmul(W, R))
    return node


def insert_node_between_two_nodes(graph: Graph,
                                  node_to_insert: BaseNode,
                                  first_node: BaseNode,
                                  last_node: BaseNode):
    graph.add_node(node_to_insert)
    e_attr = graph.get_edge_data(first_node, last_node)
    assert len(list(e_attr.values())) == 1
    e_attr = list(e_attr.values())[0]
    graph.add_edge(first_node, node_to_insert, **e_attr)
    graph.add_edge(node_to_insert, last_node, **e_attr)
    graph.remove_edge(first_node, last_node)


def insert_node_after_node(graph: Graph,
                           node_to_insert: BaseNode,
                           first_node: BaseNode):
    successors = graph.get_next_nodes(first_node)
    edge_attrs = [copy.deepcopy(graph.get_edge_data(first_node, succ)) for succ in successors]

    # Add node
    graph.add_node(node_to_insert)

    # Remove original edges
    for succ in successors:
        graph.remove_edge(first_node, succ)

    new_edge_attr = {0: {EDGE_SOURCE_INDEX: 0, EDGE_SINK_INDEX: 0}}

    # Connect first_node → node_to_insert
    graph.add_edge(first_node, node_to_insert, **new_edge_attr[0])

    # Connect node_to_insert → original successors
    for succ, attr in zip(successors, edge_attrs):
        # Use original edge attributes if available
        for key, val in attr.items():
            graph.add_edge(node_to_insert, succ, **val)


def insert_node_before_node(graph: Graph,
                            node_to_insert: BaseNode,
                            last_node: BaseNode):
    first_nodes = graph.get_prev_nodes(last_node)
    first_node = first_nodes[0]
    insert_node_between_two_nodes(graph, node_to_insert, first_node, last_node)


def insert_rotation_embed_after_first_shortcut_node(graph, linear_node, ln_node, chain_start_node, R):
    # TODO:
    # Fold!
    # get C and norm term from arg
    d_out = ln_node.weights['weight'].shape[-1]
    C = np.eye(d_out) - np.ones((d_out, d_out)) / d_out

    # add x @ R af chain_start_node
    matmul_name = f'{chain_start_node.name}_R'
    matmul_node = BaseNode(name=matmul_name,
                           framework_attr={'in_features': R.shape[0], 'out_features': R.shape[0],
                                           BIAS: False},
                           input_shape=[chain_start_node.input_shape[0]],
                           output_shape=[chain_start_node.input_shape[0]],
                           weights={'weight': np.matmul(R, C)},
                           layer_class=nn.Linear)

    matmul_node.candidates_quantization_cfg = copy.deepcopy(linear_node.candidates_quantization_cfg)
    matmul_node.prior_info = copy.deepcopy(linear_node.prior_info)
    insert_node_after_node(graph, matmul_node, chain_start_node)
    return graph


def rotate_graph(graph: Graph):
    # Find residual nodes (Add with 2inp/2out with add inp/out)
    rotationable_chains = get_rotationable_chains(graph)

    # check each block is rotatable
    for chain in rotationable_chains:
        valid_nodes = [chain[0]]
        input_nodes = []
        output_nodes = []
        norm_nodes = []
        for idx, first_node in enumerate(chain):
            if idx < len(chain) - 1:
                second_node = chain[idx + 1]
                rotationable_block, norm_node = check_block(graph, first_node, second_node)
                if len(rotationable_block) == NUM_BRANCHES:  # should be 2?
                    valid_nodes.append(second_node)
                    input_nodes.append(rotationable_block[0])
                    output_nodes.append(rotationable_block[1])
                    # TODO:
                    # make a diff mechanism for norm checking
                    # if LN:
                    # check for LN in all shortcuts + start/end of chain and apply centering
                    # if RMS:
                    # Just fold
                    # If identity: ignore
                    # BN: TBD
                    norm_nodes.append(norm_node[0])
                else:
                    valid_nodes = [second_node]
                    input_nodes = []
                    output_nodes = []

        hidden_size = valid_nodes[0].output_shape[0][-1]
        R1 = get_orthogonal_matrix(hidden_size)

        # Step 1 - Add Rotation and reduce mean (Construct Linear) after Embedding conv and positional embedding
        # TODO:
        # after checking for norms term, add it to the function, if to enable centering/folding
        # fold into the prev linear, now constructing Linear Layer
        graph = insert_rotation_embed_after_first_shortcut_node(graph, input_nodes[0], norm_nodes[0], valid_nodes[0], R1)

        # Step 2 - Iterate over each SubBlock - LN, LINEAR_IN and LINEAR_OUT
        for idx, (norm_node, in_node, out_node) in enumerate(zip(norm_nodes, input_nodes, output_nodes)):
            # Step 2.a - Fold Beta and Gamma from LN to the next Linear
            # TODO:
            # get a flag from arg
            in_node.weights['bias'] = in_node.weights['bias'] + in_node.weights['weight'] @ norm_node.weights['bias']
            norm_node.weights['bias'] = np.zeros_like(norm_node.weights['bias'])
            in_node.weights['weight'] *= norm_node.weights['weight'][np.newaxis, :]
            norm_node.weights['weight'] = np.ones_like(norm_node.weights['weight'])

            # Step 2.b
            # TODO:
            # check acc without it
            # apply from outside flag
            # norm_node.layer_class = nn.RMSNorm
            # norm_node.framework_attr.pop('bias')
            # norm_node.weights.pop('bias')

            # Step 2.c - Fold R.T into LINEAR_IN
            rotate_input_linear(in_node, R1.T)

            # Step 2.d - Fold R and reduce mean into LINEAR_OUT
            d_out = R1.shape[-1]
            C = np.eye(d_out) - np.ones((d_out, d_out)) / d_out
            rotate_output_linear(out_node, np.matmul(R1, C))

        # Step 3 - Add R.T (Construct Linear) bf LN head
        end_chain_node = graph.get_next_nodes(valid_nodes[-1])[0]
        if is_transparent(end_chain_node):
            end_chain_node = get_next_non_transparent_nodes(graph, end_chain_node)[0]
        matmul_head_name = f'{end_chain_node.name}_RT'
        matmul_head_node = BaseNode(name=matmul_head_name,
                                    framework_attr={'in_features': R1.shape[0], 'out_features': R1.shape[0],
                                                    BIAS: False},
                                    input_shape=end_chain_node.input_shape,
                                    output_shape=end_chain_node.input_shape,
                                    weights={'weight': R1.T},
                                    layer_class=nn.Linear)

        matmul_head_node.candidates_quantization_cfg = copy.deepcopy(input_nodes[0].candidates_quantization_cfg)
        matmul_head_node.prior_info = copy.deepcopy(input_nodes[0].prior_info)
        insert_node_before_node(graph, matmul_head_node, end_chain_node)
    return graph
