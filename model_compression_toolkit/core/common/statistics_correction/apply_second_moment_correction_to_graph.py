# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Callable

from tqdm import tqdm

from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_activations_computation import \
    get_activations_qparams
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute


def _assign_act_threshold(graph: Graph,
                          representative_data_gen: Callable,
                          core_config: CoreConfig,
                          fw_info: FrameworkInfo,
                          fw_impl: FrameworkImplementation):
    """
     Collect statistics after second moment correction and assign new activations thresholds.
     Args:
        graph: Graph to apply second moment correction.
        representative_data_gen (Callable): Dataset used for calibration.
        core_config (CoreConfig): Configuration object containing parameters of how the model should be
         quantized, including mixed precision parameters.
        fw_info: FrameworkInfo object with information about the specific framework's model.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
     """

    analyzer_graph(fw_impl.attach_sc_to_node,
                   graph,
                   fw_info,
                   core_config.quantization_config)  # Mark points for statistics collection

    mi = ModelCollector(graph,
                        fw_impl,
                        fw_info)

    for _ in tqdm(range(core_config.n_iter)):
        mi.infer(representative_data_gen())

    for n in list(graph.nodes):
        if n.is_activation_quantization_enabled():
            activation_params = get_activations_qparams(
                activation_quant_cfg=n.final_activation_quantization_cfg,
                nodes_prior_info=n.prior_info,
                out_stats_container=graph.get_out_stats_collector(n))
            n.final_activation_quantization_cfg.set_activation_quantization_param(activation_params)


def apply_second_moment_correction_to_graph(graph_to_apply_second_moment_correction: Graph,
                                            representative_data_gen: Callable,
                                            core_config: CoreConfig,
                                            fw_info: FrameworkInfo,
                                            fw_impl: FrameworkImplementation) -> Graph:
    """
     Apply second moment correction on graph.
     Args:
        graph_to_apply_second_moment_correction: Graph to apply second moment correction.
        representative_data_gen (Callable): Dataset used for calibration.
        core_config (CoreConfig): Configuration object containing parameters of how the model should be
         quantized, including mixed precision parameters.
        fw_info: FrameworkInfo object with information about the specific framework's model.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

     Returns:
         Graph after second moment correction.
     """
    graph = copy.deepcopy(graph_to_apply_second_moment_correction)
    semi_quantized_model = fw_impl.quantized_model_builder_for_second_moment_correction(graph, fw_info)
    fw_impl.apply_second_moment_correction(semi_quantized_model, core_config, representative_data_gen, graph)
    graph = substitute(graph, fw_impl.get_substitutions_after_second_moment_correction(core_config.quantization_config))
    _assign_act_threshold(graph, representative_data_gen, core_config, fw_info, fw_impl)

    return graph