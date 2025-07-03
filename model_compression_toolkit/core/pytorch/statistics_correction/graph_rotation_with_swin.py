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

# ViT:

def is_add(node):
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


def get_single_add_successor(graph, n):
    """Follow transparent nodes to find a single downstream add"""
    visited = set()
    frontier = list(graph.get_next_nodes(n))
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        if is_add(current):
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
    valid_users = [u for u in users if is_add(u)]
    # last
    if len(valid_users) == 0 and len(users) == 1:
        return 1
    elif len(valid_users) > 0:
        return len(users)
    else:
        return 0


def get_rotationable_nodes(graph):
    chains = []
    visited = set()

    nodes = list(graph.get_topo_sorted_nodes())
    for idx, node in enumerate(nodes):
        # if node in visited or not is_add(node):
        if node in visited or is_transparent(node):
            continue

        # Check if node has 2 users, and one is Add
        users = graph.get_next_nodes(node)
        if len(users) > 0:
            # users = [get_next_non_transparent_nodes(graph, user) if is_transparent(user) else user for user in users][0]
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
        if user_count(graph, node) != 2:
            continue

        add_users = [u for u in users if is_add(u)]
        if len(add_users) != 1:
            continue
        # Start a potential chain
        chain = [node]
        visited.add(node)

        # current = node

        current = add_users[0]
        chain.append(current)
        visited.add(current)
        while True:
            # node_inputs = graph.get_prev_nodes(current)
            node_outputs = get_single_add_successor(graph, current)

            # num_inputs = len(node_inputs)
            # add_inputs = [n for n in node_inputs if is_add(n)]
            # num_add_inputs = len(add_inputs)

            # num_outputs = len(node_outputs)
            if node_outputs is None or node_outputs in visited:
                break

            chain.append(node_outputs)
            visited.add(node_outputs)
            current = node_outputs

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
    linaer_nodes = []
    ln_nodes = []
    next_first_node = graph.get_next_nodes(first_node)
    next_first_node = [
        node
        for user in next_first_node
        for node in (
        get_next_non_transparent_nodes(graph, user)
        if is_transparent(user) else [user]
    )
    ]
    next_first_node_wo_add = [nfn for nfn in next_first_node if not is_add(nfn)]
    # if is_transparent(next_first_node_wo_add[0]):
    #     next_first_node = graph.get_next_nodes(next_first_node_wo_add[0])
    #     next_first_node_wo_add = [nfn for nfn in next_first_node if not is_add(nfn)]

    # TODO:
    # Assume only one node?
    if is_norm(next_first_node_wo_add[0]):
        ln_nodes.append(next_first_node_wo_add[0])
        next_first_node_wo_add = graph.get_next_nodes(next_first_node_wo_add[0])
        next_first_node_wo_add = [
                node
                for user in next_first_node_wo_add
                for node in (
                    get_next_non_transparent_nodes(graph, user)
                    if is_transparent(user) else [user]
                )
            ]

        # TODO:
        # Assume only one node?
        if is_linear(next_first_node_wo_add[0]):
            linaer_nodes.append(next_first_node_wo_add[0])

        prev_second_node = graph.get_prev_nodes(second_node)
        prev_second_node_wo_add = [
                node
                for user in prev_second_node
                for node in (
                    get_prev_non_transparent_nodes(graph, user)
                    if is_transparent(user) else [user]
                )
            ]
        if first_node in prev_second_node_wo_add:
            prev_second_node_wo_add.remove(first_node)
        # prev_second_node_wo_add = [psn for psn in prev_second_node_wo_add if not is_add(psn)]

        # TODO:
        # Assume only one node?
        if is_linear(prev_second_node_wo_add[0]):
            linaer_nodes.append(prev_second_node_wo_add[0])
        # prev_nodes = []
        # for node in prev_second_node_wo_add:
        #     if is_transparent(node):
        #         temp_nodes = (graph.get_prev_nodes(node))
        #         temp_nodes_no_add = [tn for tn in temp_nodes if not is_add(tn)]
        #         prev_nodes.append(temp_nodes_no_add)
        #     else:
        #         prev_nodes.append(node)
        # if prev_nodes and all(isinstance(sublist, list) for sublist in prev_nodes):
        #     prev_nodes = [item for sublist in prev_nodes for item in sublist]
        #
        # if is_linear(prev_nodes[0]):
        #     linaer_nodes.append(prev_nodes[0])
    else:
        Logger.warning("No LN")
    return linaer_nodes, ln_nodes


def random_orthogonal_matrix(size):
    # torch.cuda.empty_cache()
    # random_matrix = torch.randn(size, size)
    # q, r = torch.linalg.qr(random_matrix)
    # q *= torch.sign(torch.diag(r)).unsqueeze(0)
    # return q
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


def rotate_output_conv(node, R):
    # conv.weight: [out_channels, in_channels, kH, kW]
    # Apply R to the output channel dimension (dim=0)
    new_weight = np.einsum('oi,ihwk->ohwk', R, node.weights['weight'])

    node.set_weights_by_keys('weight', new_weight)

    # Also fold into bias if it exists
    if 'bias' in node.weights.keys() and node.weights['bias'] is not None:
        node.set_weights_by_keys('bias', R @ node.weights['bias'])

    return node


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
    """
    Insert a new node in a graph between two nodes.

    Args:
        graph: Graph to add the new node to.
        node_to_insert: Node to add.
        first_node: Node to insert the new node after it.
        last_node: Node to insert the new node before it.

    """

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
    """
    Insert a new node to a graph after an existing node in the graph.
    Check before insertion that the node (that we add the new node after) has
    only a single outgoing edge, so such an insertion is possible. If it is not the
    case, an exception is thrown.

    Args:
        graph: Graph to add the new node to.
        node_to_insert: Node to add.
        first_node: Node to insert the new node after it.

    """

    # last_nodes = graph.get_next_nodes(first_node)
    # # if len(last_nodes) != 1:
    # #     Logger.critical(
    # #         f'Insertion requires exactly one successor node; {len(last_nodes)} successors found.')  # pragma: no cover
    # last_node = last_nodes[0]
    # insert_node_between_two_nodes(graph, node_to_insert, first_node, last_node)
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
    """
    Insert a new node to a graph before an existing node in the graph.
    Check before insertion that the node (that we add the new node before) has
    only a single incoming edge, so such an insertion is possible. If it is not the
    case, an exception is thrown.

    Args:
        graph: Graph to add the new node to.
        node_to_insert: Node to add.
        last_node: Node to insert the new node after it.

    """
    first_nodes = graph.get_prev_nodes(last_node)
    # if len(first_nodes) != 1:
    #     Logger.critical('Insertion requires exactly one predecessor node; multiple or no predecessors found.')  # pragma: no cover
    first_node = first_nodes[0]
    insert_node_between_two_nodes(graph, node_to_insert, first_node, last_node)


def insert_rotation_embed(graph, ln_node, add_node, R):
    quant_candidate = copy.deepcopy(ln_node.candidates_quantization_cfg[0])
    quant_candidate.activation_quantization_cfg.quant_mode = quant_candidate.activation_quantization_cfg.quant_mode.NO_QUANT
    quant_candidate.weights_quantization_cfg.disable_all_weights_quantization()
    # add x @ R bf add
    matmul_name = f'{add_node.name}_R'
    matmul_node = BaseNode(name=matmul_name,
                           framework_attr={'in_features': R.shape[0], 'out_features': R.shape[0],
                                           BIAS: False},
                           input_shape=add_node.input_shape[0],
                           output_shape=add_node.input_shape[0],
                           weights={'weight': R},
                           layer_class=nn.Linear)
    matmul_node.candidates_quantization_cfg = [quant_candidate]
    insert_node_before_node(graph, matmul_node, add_node)
    add_node.set_weights_by_keys(1, add_node.weights[1] @ R)
    return graph


def insert_rotation_embed_after_add(graph, linear_node, ln_node, add_node, R):

    d_out = ln_node.weights['weight'].shape[-1]
    C = np.eye(d_out) - np.ones((d_out, d_out)) / d_out

    # quant_candidate = copy.deepcopy(ln_node.candidates_quantization_cfg[0])
    # quant_candidate.activation_quantization_cfg.quant_mode = quant_candidate.activation_quantization_cfg.quant_mode.NO_QUANT
    # quant_candidate.weights_quantization_cfg.disable_all_weights_quantization()
    # add x @ R bf add
    matmul_name = f'{add_node.name}_R'
    matmul_node = BaseNode(name=matmul_name,
                           framework_attr={'in_features': R.shape[0], 'out_features': R.shape[0],
                                           BIAS: False},
                           input_shape=[add_node.input_shape[0]],
                           output_shape=[add_node.input_shape[0]],
                           weights={'weight': np.matmul(R, C)},
                           layer_class=nn.Linear)
    # matmul_node.candidates_quantization_cfg = [quant_candidate]
    matmul_node.candidates_quantization_cfg = copy.deepcopy(linear_node.candidates_quantization_cfg)
    matmul_node.prior_info = copy.deepcopy(linear_node.prior_info)
    insert_node_after_node(graph, matmul_node, add_node)
    # add_node.set_weights_by_keys(1, add_node.weights[1]@R)
    # add_node.candidates_quantization_cfg[0].activation_quantization_cfg.quant_mode = \
    # add_node.candidates_quantization_cfg[0].activation_quantization_cfg.quant_mode.NO_QUANT
    # add_node.candidates_quantization_cfg[0].weights_quantization_cfg.disable_all_weights_quantization()
    return graph


def wrap_ln(graph, ln_nodes, R):
    quant_candidate = copy.deepcopy(ln_nodes[0].candidates_quantization_cfg[0])
    quant_candidate.activation_quantization_cfg.quant_mode = quant_candidate.activation_quantization_cfg.quant_mode.NO_QUANT
    quant_candidate.weights_quantization_cfg.disable_all_weights_quantization()
    for ln_node in ln_nodes:
        # add x @ R.T bf LN
        matmul_name = f'{ln_node.name}_RT'
        # matmul_node = FunctionalNode(name=matmul_name,
        #                              framework_attr={},
        #                              input_shape=ln_node.input_shape,
        #                              output_shape=ln_node.output_shape,
        #                              weights={},
        #                              layer_class=torch.matmul,
        #                              op_call_args=[],
        #                              op_call_kwargs={},
        #                              functional_op=torch.matmul)
        matmul_node = BaseNode(name=matmul_name,
                               framework_attr={'in_features': R.shape[0], 'out_features': R.shape[0],
                                               BIAS: False},
                               input_shape=ln_node.input_shape,
                               output_shape=ln_node.output_shape,
                               weights={'weight': R.T},
                               layer_class=nn.Linear)
        matmul_node.candidates_quantization_cfg = [quant_candidate]
        insert_node_before_node(graph, matmul_node, ln_node)

        matmul_out_name = f'{ln_node.name}_R'
        matmul_out_node = BaseNode(name=matmul_out_name,
                                   framework_attr={'in_features': R.shape[0], 'out_features': R.shape[0],
                                                   BIAS: False},
                                   input_shape=ln_node.input_shape,
                                   output_shape=ln_node.output_shape,
                                   weights={'weight': R},
                                   layer_class=nn.Linear)
        matmul_out_node.candidates_quantization_cfg = [quant_candidate]
        insert_node_after_node(graph, matmul_out_node, ln_node)
        ln_node.candidates_quantization_cfg = [quant_candidate]

    return graph


def rotate_graph_old(graph: Graph,
                 fw_impl: FrameworkImplementation):
    # Find residual nodes (Add with 2inp/2out with add inp/out)
    # Assume only one chain
    rotationable_nodes = get_rotationable_nodes(graph)[0]
    # check each block is rotatable
    valid_nodes = [rotationable_nodes[0]]
    input_nodes = []
    output_nodes = []
    ln_nodes = []
    for idx, first_node in enumerate(rotationable_nodes):
        if idx < len(rotationable_nodes) - 1:
            second_node = rotationable_nodes[idx + 1]
            rotationable_block, ln_node = check_block(graph, first_node, second_node)
            if len(rotationable_block) == 2:  # should be 2?
                valid_nodes.append(second_node)
                input_nodes.append(rotationable_block[0])
                output_nodes.append(rotationable_block[1])
                ln_nodes.append(ln_node[0])
            else:
                valid_nodes = [second_node]
                input_nodes = []
                output_nodes = []

    embed_node = None
    # Get input/output conv (very vit oriented)
    for node in graph.get_topo_sorted_nodes():
        if node.type in [torch.nn.Conv2d, torch.nn.Linear]:
            if embed_node is None:
                embed_node = node
            head_node = node
    # fold LN/RMSN (move into general substitution later)
    # Calc rotation matrix
    hidden_size = valid_nodes[0].output_shape[0][-1]
    R1 = get_orthogonal_matrix(hidden_size)

    # quant_candidate = copy.deepcopy(ln_nodes[0].candidates_quantization_cfg[0])
    # quant_candidate.activation_quantization_cfg.quant_mode = quant_candidate.activation_quantization_cfg.quant_mode.NO_QUANT
    # quant_candidate.weights_quantization_cfg.disable_all_weights_quantization()

    # TODO:
    # check because follow spinquant paper but the code is different and strange
    # Insert Linear bf first add for now

    # Step 1 - Add Rotation and reduce mean (Construct Linear) after Embedding conv and positional embedding
    graph = insert_rotation_embed_after_add(graph, input_nodes[0], ln_nodes[0], valid_nodes[0], R1)

    # Step 2 - Iterate over each SubBlock - LN, LINEAR_IN and LINEAR_OUT
    for idx, (ln_node, in_node, out_node) in enumerate(zip(ln_nodes, input_nodes, output_nodes)):

        # Step 2.a - Wrap each LN with R.T (Construct Linear) bf and R af (Construct Linear)

        # add x @ R.T bf LN
        # matmul_name = f'{ln_node.name}_RT'
        # matmul_node = BaseNode(name=matmul_name,
        #                        framework_attr={'in_features': R1.shape[0], 'out_features': R1.shape[0],
        #                                        BIAS: False},
        #                        input_shape=ln_node.input_shape,
        #                        output_shape=ln_node.output_shape,
        #                        weights={'weight': R1.T},
        #                        layer_class=nn.Linear)
        # matmul_node.candidates_quantization_cfg = [quant_candidate]
        # insert_node_before_node(graph, matmul_node, ln_node)

        # # add reduce mean bf LN
        # d_out = ln_node.weights['weight'].shape[-1]
        # C = np.eye(d_out) - np.ones((d_out, d_out)) / d_out
        # reduce_mean_name = f'{ln_node.name}_RM'
        # reduce_mean_node = BaseNode(name=reduce_mean_name,
        #                             framework_attr={'in_features': R1.shape[0], 'out_features': R1.shape[0],
        #                                             BIAS: False},
        #                             input_shape=ln_node.input_shape,
        #                             output_shape=ln_node.output_shape,
        #                             weights={'weight': C},
        #                             layer_class=nn.Linear)
        # reduce_mean_node.candidates_quantization_cfg = [quant_candidate]
        # insert_node_before_node(graph, reduce_mean_node, ln_node)

        # add x @ R af LN
        # matmul_out_name = f'{ln_node.name}_R'
        # matmul_out_node = BaseNode(name=matmul_out_name,
        #                            framework_attr={'in_features': R1.shape[0], 'out_features': R1.shape[0],
        #                                            BIAS: False},
        #                            input_shape=ln_node.input_shape,
        #                            output_shape=ln_node.output_shape,
        #                            weights={'weight': R1},
        #                            layer_class=nn.Linear)
        # matmul_out_node.candidates_quantization_cfg = [quant_candidate]
        # insert_node_after_node(graph, matmul_out_node, ln_node)

        # ln_node.candidates_quantization_cfg = [quant_candidate]

        # ADD R.T bf first shortcut
        # second_add_node = valid_nodes[1]
        # # prev_second_add_node = graph.get_prev_nodes(second_add_node)[0]
        # matmul_out_name = f'{second_add_node.name}_RT'
        # matmul_out_node = BaseNode(name=matmul_out_name,
        #                            framework_attr={'in_features': R1.shape[0], 'out_features': R1.shape[0],
        #                                            BIAS: False},
        #                            input_shape=second_add_node.output_shape,
        #                            output_shape=second_add_node.output_shape,
        #                            weights={'weight': R1.T},
        #                            layer_class=nn.Linear)
        # matmul_out_node.candidates_quantization_cfg = [quant_candidate]
        # insert_node_before_node(graph, matmul_out_node, second_add_node)

        # for idx, (in_node, out_node) in enumerate(zip(input_nodes, output_nodes)):
        # add x @ R af out Linear
        # matmul_out_name = f'{out_node.name}_RT'
        # matmul_out_node = BaseNode(name=matmul_out_name,
        #                            framework_attr={'in_features': R1.shape[0], 'out_features': R1.shape[0],
        #                                            BIAS: False},
        #                            input_shape=out_node.output_shape,
        #                            output_shape=out_node.output_shape,
        #                            weights={'weight': R1},
        #                            layer_class=nn.Linear)
        # matmul_out_node.candidates_quantization_cfg = [quant_candidate]
        # insert_node_after_node(graph, matmul_out_node, out_node)

        # Step 2.b - Fold Beta and Gamma from LN to the next Linear
        in_node.weights['bias'] = in_node.weights['bias'] + in_node.weights['weight'] @ ln_node.weights['bias']
        ln_node.weights['bias'] = np.zeros_like(ln_node.weights['bias'])
        in_node.weights['weight'] *= ln_node.weights['weight'][np.newaxis, :]
        ln_node.weights['weight'] = np.ones_like(ln_node.weights['weight'])

        # Step 2.bb
        ln_node.layer_class = nn.RMSNorm
        ln_node.framework_attr.pop('bias')
        ln_node.weights.pop('bias')

        # Step 2.c - Fold R.T into LINEAR_IN
        rotate_input_linear(in_node, R1.T)

        # Step 2.d - Fold R and reduce mean into LINEAR_OUT
        d_out = R1.shape[-1]
        C = np.eye(d_out) - np.ones((d_out, d_out)) / d_out
        rotate_output_linear(out_node, np.matmul(R1, C))

    # Step 3 - Add R.T (Construct Linear) bf LN head
    head_norm = graph.get_next_nodes(valid_nodes[-1])[0]
    matmul_head_name = f'{head_norm.name}_RT'
    matmul_head_node = BaseNode(name=matmul_head_name,
                                framework_attr={'in_features': R1.shape[0], 'out_features': R1.shape[0],
                                                BIAS: False},
                                input_shape=head_norm.input_shape,
                                output_shape=head_norm.input_shape,
                                weights={'weight': R1.T},
                                layer_class=nn.Linear)
    # matmul_head_node.candidates_quantization_cfg = [quant_candidate]
    matmul_head_node.candidates_quantization_cfg = copy.deepcopy(input_nodes[0].candidates_quantization_cfg)
    matmul_head_node.prior_info = copy.deepcopy(input_nodes[0].prior_info)
    insert_node_before_node(graph, matmul_head_node, head_norm)

    # head_norm.candidates_quantization_cfg = [quant_candidate]

    return graph


def rotate_graph(graph: Graph):
    # Find residual nodes (Add with 2inp/2out with add inp/out)
    # Assume only one chain
    rotationable_nodes = get_rotationable_nodes(graph)
    # check each block is rotatable
    for chain in rotationable_nodes:
        valid_nodes = [chain[0]]
        input_nodes = []
        output_nodes = []
        ln_nodes = []
        for idx, first_node in enumerate(chain):
            if idx < len(chain) - 1:
                second_node = chain[idx + 1]
                rotationable_block, ln_node = check_block(graph, first_node, second_node)
                if len(rotationable_block) == 2:  # should be 2?
                    valid_nodes.append(second_node)
                    input_nodes.append(rotationable_block[0])
                    output_nodes.append(rotationable_block[1])
                    ln_nodes.append(ln_node[0])
                else:
                    valid_nodes = [second_node]
                    input_nodes = []
                    output_nodes = []

        # Check last shortcut for LN
        # maybe add another check for LN in each shortcut?
        last_ln = graph.get_next_nodes(chain[-1])
        if is_transparent(last_ln[0]):
            last_ln = get_next_non_transparent_nodes(graph, last_ln[0])
        # if len(last_ln) == 1 and is_norm(last_ln[0]):
        if len(last_ln) == 1:
            # embed_node = None
            # # Get input/output conv (very vit oriented)
            # for node in graph.get_topo_sorted_nodes():
            #     if node.type in [torch.nn.Conv2d, torch.nn.Linear]:
            #         if embed_node is None:
            #             embed_node = node
            #         head_node = node
            # fold LN/RMSN (move into general substitution later)
            # Calc rotation matrix
            hidden_size = valid_nodes[0].output_shape[0][-1]
            R1 = get_orthogonal_matrix(hidden_size)

            # Step 1 - Add Rotation and reduce mean (Construct Linear) after Embedding conv and positional embedding
            graph = insert_rotation_embed_after_add(graph, input_nodes[0], ln_nodes[0], valid_nodes[0], R1)

            # Step 2 - Iterate over each SubBlock - LN, LINEAR_IN and LINEAR_OUT
            for idx, (ln_node, in_node, out_node) in enumerate(zip(ln_nodes, input_nodes, output_nodes)):

                # Step 2.a - Fold Beta and Gamma from LN to the next Linear
                in_node.weights['bias'] = in_node.weights['bias'] + in_node.weights['weight'] @ ln_node.weights['bias']
                ln_node.weights['bias'] = np.zeros_like(ln_node.weights['bias'])
                in_node.weights['weight'] *= ln_node.weights['weight'][np.newaxis, :]
                ln_node.weights['weight'] = np.ones_like(ln_node.weights['weight'])

                # Step 2.b
                # ln_node.layer_class = nn.RMSNorm
                # ln_node.framework_attr.pop('bias')
                # ln_node.weights.pop('bias')

                # Step 2.c - Fold R.T into LINEAR_IN
                rotate_input_linear(in_node, R1.T)

                # Step 2.d - Fold R and reduce mean into LINEAR_OUT
                d_out = R1.shape[-1]
                C = np.eye(d_out) - np.ones((d_out, d_out)) / d_out
                rotate_output_linear(out_node, np.matmul(R1, C))

            # Step 3 - Add R.T (Construct Linear) bf LN head
            head_norm = graph.get_next_nodes(valid_nodes[-1])[0]
            if is_transparent(head_norm):
                head_norm = get_next_non_transparent_nodes(graph, head_norm)[0]
            matmul_head_name = f'{head_norm.name}_RT'
            matmul_head_node = BaseNode(name=matmul_head_name,
                                        framework_attr={'in_features': R1.shape[0], 'out_features': R1.shape[0],
                                                        BIAS: False},
                                        input_shape=head_norm.input_shape,
                                        output_shape=head_norm.input_shape,
                                        weights={'weight': R1.T},
                                        layer_class=nn.Linear)

            matmul_head_node.candidates_quantization_cfg = copy.deepcopy(input_nodes[0].candidates_quantization_cfg)
            matmul_head_node.prior_info = copy.deepcopy(input_nodes[0].prior_info)
            insert_node_before_node(graph, matmul_head_node, head_norm)
        else:
            Logger.warning("No LN")
    return graph
