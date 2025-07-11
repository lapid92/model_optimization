# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
import unittest

import keras
import numpy as np

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.pruning.memory_calculator import MemoryCalculator
from model_compression_toolkit.core.keras.pruning.pruning_keras_implementation import PruningKerasImplementation
from model_compression_toolkit.core.graph_prep_runner import read_model_to_graph

from model_compression_toolkit.quantization_preparation.load_fqc import load_fqc_configuration
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras

layers = keras.layers


class TestParameterCounter(unittest.TestCase):
    # TODO: Extend it to more layers and scenarios

    def representative_dataset(self, in_shape=(1,8,8,3)):
        for _ in range(1):
            yield [np.random.randn(*in_shape)]

    def test_conv_layer(self):
        # Define the layer
        out_channels = 2
        in_channels = 1
        kernel_size = 3
        use_bias=True

        inputs = layers.Input(shape=(8, 8, in_channels))
        x = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, use_bias=use_bias)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        fw_impl = PruningKerasImplementation()
        tpc = mct.get_target_platform_capabilities('tensorflow', 'imx500')

        fqc = AttachTpcToKeras().attach(tpc)

        # Convert the original Keras model to an internal graph representation.
        float_graph = read_model_to_graph(model,
                                          self.representative_dataset,
                                          fqc,
                                          fw_impl)

        # Apply quantization configuration to the graph. This step is necessary even when not quantizing,
        # as it prepares the graph for the pruning process.
        float_graph_with_compression_config = load_fqc_configuration(float_graph, fqc)

        self.memory_calculator = MemoryCalculator(graph=float_graph_with_compression_config,
                                                  fw_impl=fw_impl)

        # masks = {list(float_graph_with_compression_config.nodes)[0]}
        counted_params = self.memory_calculator.get_pruned_graph_num_params(masks=None,
                                                                            include_padded_channels=tpc.is_simd_padding)

        # Calculate expected number of parameters
        simd_groups = np.ceil(out_channels/32.)
        expected_params = 32 * simd_groups * (in_channels * kernel_size * kernel_size + int(use_bias))
        self.assertEqual(counted_params, expected_params)

        counted_params = self.memory_calculator.get_pruned_graph_num_params(masks=None,
                                                                            include_padded_channels=False)

        # Calculate expected number of parameters
        expected_params = out_channels * (in_channels * kernel_size * kernel_size + int(use_bias))
        self.assertEqual(counted_params, expected_params)
