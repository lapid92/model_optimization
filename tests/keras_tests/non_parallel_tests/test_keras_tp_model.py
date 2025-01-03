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
import keras
import unittest
from functools import partial

import numpy as np
import tensorflow as tf
from packaging import version

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.core.common import BaseNode
from tests.common_tests.helpers.generate_test_tp_model import generate_test_op_qc, generate_test_attr_configs

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, Conv2DTranspose, ReLU, Activation, BatchNormalization
    from keras.src import Input
else:
    from keras.layers import Conv2D, Conv2DTranspose, ReLU, Activation, BatchNormalization
    from keras import Input

import model_compression_toolkit as mct
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import \
    LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.attribute_filter import \
    Greater, \
    Smaller, GreaterEq, Eq, SmallerEq, Contains
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL, IMX500_TP_MODEL, \
    QNNPACK_TP_MODEL, TFLITE_TP_MODEL, KERNEL_ATTR, BIAS_ATTR, KERAS_KERNEL, BIAS, WEIGHTS_N_BITS
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation

tp = mct.target_platform

TEST_QC = generate_test_op_qc(**generate_test_attr_configs())
TEST_QCO = schema.QuantizationConfigOptions(tuple([TEST_QC]))


def get_node(layer) -> BaseNode:
    i = Input(shape=(3, 16, 16))
    x = layer(i)
    model = tf.keras.Model(i, x)
    graph = KerasImplementation().model_reader(model, None)
    return graph.get_topo_sorted_nodes()[1]


class TestKerasTPModel(unittest.TestCase):

    def test_keras_layers_with_params(self):
        conv_with_params = LayerFilterParams(Conv2D,
                                             Greater("filters", 2),
                                             Smaller("filters", 4),
                                             activation='softmax',
                                             kernel_size=(3, 4),
                                             filters=3)

        conv = Conv2D(filters=3, kernel_size=(3, 4), activation='softmax')
        self.assertTrue(get_node(conv).is_match_filter_params(conv_with_params))
        conv = Conv2D(filters=2, kernel_size=(3, 4), activation='softmax')
        self.assertFalse(get_node(conv).is_match_filter_params(conv_with_params))
        conv = Conv2DTranspose(filters=3, kernel_size=(3, 4), activation='softmax')
        self.assertFalse(get_node(conv).is_match_filter_params(conv_with_params))

        relu_with_params = LayerFilterParams(ReLU, GreaterEq("max_value", 0.5) | Smaller("max_value", 0.2))
        self.assertTrue(get_node(ReLU(max_value=0.1)).is_match_filter_params(relu_with_params))
        self.assertTrue(get_node(ReLU(max_value=0.5)).is_match_filter_params(relu_with_params))
        self.assertFalse(get_node(ReLU(max_value=0.3)).is_match_filter_params(relu_with_params))

        relu_with_params = LayerFilterParams(ReLU, Eq("max_value", None) | Eq("max_value", 6))
        self.assertTrue(get_node(ReLU()).is_match_filter_params(relu_with_params))
        self.assertTrue(get_node(ReLU(max_value=6)).is_match_filter_params(relu_with_params))
        self.assertFalse(get_node(ReLU(max_value=8)).is_match_filter_params(relu_with_params))

        lrelu_with_params = LayerFilterParams(tf.nn.leaky_relu, SmallerEq("alpha", 2))
        self.assertTrue(get_node(partial(tf.nn.leaky_relu, alpha=0.4)).is_match_filter_params(lrelu_with_params))
        self.assertTrue(get_node(partial(tf.nn.leaky_relu, alpha=2)).is_match_filter_params(lrelu_with_params))
        self.assertFalse(get_node(partial(tf.nn.leaky_relu, alpha=2.1)).is_match_filter_params(lrelu_with_params))

        lrelu_with_params = LayerFilterParams(tf.nn.leaky_relu)
        self.assertTrue(get_node(partial(tf.nn.leaky_relu, alpha=0.4)).is_match_filter_params(lrelu_with_params))

        conv_filter_contains = LayerFilterParams(Conv2D, Contains("name", "conv"))
        conv = Conv2D(filters=3, kernel_size=(3, 4), name="conv")
        self.assertTrue(get_node(conv).is_match_filter_params(conv_filter_contains))
        conv = Conv2D(filters=3, kernel_size=(3, 4), name="layer_conv_0")
        self.assertTrue(get_node(conv).is_match_filter_params(conv_filter_contains))
        conv = Conv2D(filters=2, kernel_size=(3, 4), name="CONVOLUTION")
        self.assertFalse(get_node(conv).is_match_filter_params(conv_filter_contains))

    def test_get_layers_by_op(self):
        op_obj = schema.OperatorsSet('opsetA')

        hm = schema.TargetPlatformModel(
            schema.QuantizationConfigOptions(tuple([TEST_QC])),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            operator_set=tuple([op_obj]),
            add_metadata=False)
        fw_tp = TargetPlatformCapabilities(hm)
        with fw_tp:
            opset_layers = [Conv2D, LayerFilterParams(ReLU, max_value=2)]
            tp.OperationsSetToLayers('opsetA', opset_layers)
        self.assertEqual(fw_tp.get_layers_by_opset_name('opsetA'), opset_layers)
        self.assertEqual(fw_tp.get_layers_by_opset(op_obj), opset_layers)
        self.assertEqual(fw_tp.get_layers_by_opset_name('nonExistingOpsetName'), None)

    def test_get_layers_by_opconcat(self):
        op_obj_a = schema.OperatorsSet('opsetA')
        op_obj_b = schema.OperatorsSet('opsetB')
        op_concat = schema.OperatorSetConcat([op_obj_a, op_obj_b])
        hm = schema.TargetPlatformModel(
            schema.QuantizationConfigOptions(tuple([TEST_QC])),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            operator_set=tuple([op_obj_a, op_obj_b]),
            add_metadata=False)

        fw_tp = TargetPlatformCapabilities(hm)
        with fw_tp:
            opset_layers_a = [Conv2D]
            opset_layers_b = [LayerFilterParams(ReLU, max_value=2)]
            tp.OperationsSetToLayers('opsetA', opset_layers_a)
            tp.OperationsSetToLayers('opsetB', opset_layers_b)

        self.assertEqual(fw_tp.get_layers_by_opset(op_concat), opset_layers_a + opset_layers_b)

    def test_layer_attached_to_multiple_opsets(self):
        hm = schema.TargetPlatformModel(
            schema.QuantizationConfigOptions(tuple([TEST_QC])),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            operator_set=tuple([schema.OperatorsSet('opsetA'),
                          schema.OperatorsSet('opsetB')]),
            add_metadata=False)


        fw_tp = TargetPlatformCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with fw_tp:
                tp.OperationsSetToLayers('opsetA', [Conv2D])
                tp.OperationsSetToLayers('opsetB', [Conv2D])
        self.assertEqual('Found layer Conv2D in more than one OperatorsSet', str(e.exception))

    def test_filter_layer_attached_to_multiple_opsets(self):
        hm = schema.TargetPlatformModel(
            schema.QuantizationConfigOptions(tuple([TEST_QC])),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            operator_set=tuple([schema.OperatorsSet('opsetA'),
                          schema.OperatorsSet('opsetB')]),
            add_metadata=False)
        fw_tp = TargetPlatformCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with fw_tp:
                tp.OperationsSetToLayers('opsetA', [LayerFilterParams(Activation, activation="relu")])
                tp.OperationsSetToLayers('opsetB', [LayerFilterParams(Activation, activation="relu")])
        self.assertEqual('Found layer Activation(activation=relu) in more than one OperatorsSet', str(e.exception))

    def test_qco_by_keras_layer(self):
        operator_set = []
        default_qco = schema.QuantizationConfigOptions(tuple([TEST_QC]))
        default_qco = default_qco.clone_and_edit(attr_weights_configs_mapping={})
        mixed_precision_configuration_options = schema.QuantizationConfigOptions(
            quantization_configurations=tuple([TEST_QC,
                                      TEST_QC.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 4}}),
                                      TEST_QC.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 2}})]),
            base_config=TEST_QC)

        operator_set.append(schema.OperatorsSet("conv", mixed_precision_configuration_options))
        sevenbit_qco = TEST_QCO.clone_and_edit(activation_n_bits=7,
                                               attr_weights_configs_mapping={})
        operator_set.append(schema.OperatorsSet("tanh", sevenbit_qco))
        operator_set.append(schema.OperatorsSet("relu"))

        tpm = schema.TargetPlatformModel(default_qco,
                                         tpc_minor_version=None,
                                         tpc_patch_version=None,
                                         tpc_platform_type=None,
                                         operator_set=tuple(operator_set),
                                         add_metadata=False,
                                         name='test')

        tpc_keras = tp.TargetPlatformCapabilities(tpm)
        with tpc_keras:
            tp.OperationsSetToLayers("conv", [Conv2D],
                                     attr_mapping={KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                                                   BIAS_ATTR: DefaultDict(default_value=BIAS)})
            tp.OperationsSetToLayers("tanh", [tf.nn.tanh])
            tp.OperationsSetToLayers("relu", [LayerFilterParams(Activation, activation="relu")])

        conv_node = get_node(Conv2D(1, 1))
        tanh_node = get_node(tf.nn.tanh)
        relu_node = get_node(Activation('relu'))

        conv_qco = conv_node.get_qco(tpc_keras)
        tanh_qco = tanh_node.get_qco(tpc_keras)
        relu_qco = relu_node.get_qco(tpc_keras)

        self.assertEqual(len(conv_qco.quantization_configurations),
                         len(mixed_precision_configuration_options.quantization_configurations))
        for i in range(len(conv_qco.quantization_configurations)):
            self.assertEqual(conv_qco.quantization_configurations[i].attr_weights_configs_mapping[KERAS_KERNEL],
                             mixed_precision_configuration_options.quantization_configurations[
                                 i].attr_weights_configs_mapping[KERNEL_ATTR])
        self.assertEqual(tanh_qco, sevenbit_qco)
        self.assertEqual(relu_qco, default_qco)

    def test_opset_not_in_tp(self):
        default_qco = schema.QuantizationConfigOptions(tuple([TEST_QC]))
        hm = schema.TargetPlatformModel(default_qco,
                                        tpc_minor_version=None,
                                        tpc_patch_version=None,
                                        tpc_platform_type=None,
                                        operator_set=tuple([schema.OperatorsSet("opA")]),
                                        add_metadata=False)
        hm_keras = tp.TargetPlatformCapabilities(hm)
        with self.assertRaises(Exception) as e:
            with hm_keras:
                tp.OperationsSetToLayers("conv", [Conv2D])
        self.assertEqual(
            'conv is not defined in the target platform model that is associated with the target platform capabilities.',
            str(e.exception))

    def test_keras_fusing_patterns(self):
        default_qco = schema.QuantizationConfigOptions(tuple([TEST_QC]))
        a = schema.OperatorsSet("opA")
        b = schema.OperatorsSet("opB")
        c = schema.OperatorsSet("opC")
        operator_set = [a, b, c]
        fusing_patterns = [schema.Fusing((a, b, c)),
                           schema.Fusing((a, c))]

        hm = schema.TargetPlatformModel(default_qco,
                                        tpc_minor_version=None,
                                        tpc_patch_version=None,
                                        tpc_platform_type=None,
                                        operator_set=tuple(operator_set),
                                        fusing_patterns=tuple(fusing_patterns),
                                        add_metadata=False)

        hm_keras = tp.TargetPlatformCapabilities(hm)
        with hm_keras:
            tp.OperationsSetToLayers("opA", [Conv2D])
            tp.OperationsSetToLayers("opB", [tf.nn.tanh])
            tp.OperationsSetToLayers("opC", [LayerFilterParams(ReLU, Greater("max_value", 7), negative_slope=0)])

        fusings = hm_keras.get_fusing_patterns()
        self.assertEqual(len(fusings), 2)
        p0, p1 = fusings[0], fusings[1]

        self.assertEqual(len(p0), 3)
        self.assertEqual(p0[0], Conv2D)
        self.assertEqual(p0[1], tf.nn.tanh)
        self.assertEqual(p0[2], LayerFilterParams(ReLU, Greater("max_value", 7), negative_slope=0))

        self.assertEqual(len(p1), 2)
        self.assertEqual(p1[0], Conv2D)
        self.assertEqual(p1[1], LayerFilterParams(ReLU, Greater("max_value", 7), negative_slope=0))

    def test_get_default_op_qc(self):
        default_qco = schema.QuantizationConfigOptions(tuple([TEST_QC]))
        tpm = schema.TargetPlatformModel(default_qco,
                                         tpc_minor_version=None,
                                         tpc_patch_version=None,
                                         tpc_platform_type=None,
                                         operator_set=tuple([schema.OperatorsSet("opA")]),
                                         add_metadata=False)

        tpc = tp.TargetPlatformCapabilities(tpm)
        with tpc:
            tp.OperationsSetToLayers("opA", [Conv2D])

        d_qco = tpc.get_default_op_qc()
        self.assertEqual(d_qco, TEST_QC)


class TestGetKerasTPC(unittest.TestCase):
    def test_get_keras_tpc(self):
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)
        input_shape = (1, 8, 8, 3)
        input_tensor = Input(shape=input_shape[1:])
        conv = Conv2D(3, 3)(input_tensor)
        bn = BatchNormalization()(conv)
        relu = ReLU()(bn)
        model = keras.Model(inputs=input_tensor, outputs=relu)

        def rep_data():
            yield [np.random.randn(*input_shape)]

        quantized_model, _ = mct.ptq.keras_post_training_quantization(model,
                                                                      rep_data,
                                                                      target_platform_capabilities=tpc)
        core_config = mct.core.CoreConfig(
            mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=2,
                                                                             use_hessian_based_scores=False))
        quantized_model, _ = mct.ptq.keras_post_training_quantization(model,
                                                                      rep_data,
                                                                      core_config=core_config,
                                                                      target_resource_utilization=mct.core.ResourceUtilization(
                                                                          np.inf),
                                                                      target_platform_capabilities=tpc)

    def test_get_keras_supported_version(self):
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)  # Latest
        self.assertTrue(tpc.tp_model.tpc_minor_version == 1)

        tpc = mct.get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL, 'v1_pot')
        self.assertTrue(tpc.tp_model.tpc_minor_version == 1)
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL, 'v1_lut')
        self.assertTrue(tpc.tp_model.tpc_minor_version == 1)
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL, 'v1')
        self.assertTrue(tpc.tp_model.tpc_minor_version == 1)
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL, 'v2_lut')
        self.assertTrue(tpc.tp_model.tpc_minor_version == 2)
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL, 'v2')
        self.assertTrue(tpc.tp_model.tpc_minor_version == 2)

        tpc = mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, "v1")
        self.assertTrue(tpc.tp_model.tpc_minor_version == 1)
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, "v2")
        self.assertTrue(tpc.tp_model.tpc_minor_version == 2)

        tpc = mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, "v1_lut")
        self.assertTrue(tpc.tp_model.tpc_minor_version == 1)
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, "v2_lut")
        self.assertTrue(tpc.tp_model.tpc_minor_version == 2)

        tpc = mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, "v1_pot")
        self.assertTrue(tpc.tp_model.tpc_minor_version == 1)

        tpc = mct.get_target_platform_capabilities(TENSORFLOW, TFLITE_TP_MODEL, "v1")
        self.assertTrue(tpc.tp_model.tpc_minor_version == 1)

        tpc = mct.get_target_platform_capabilities(TENSORFLOW, QNNPACK_TP_MODEL, "v1")
        self.assertTrue(tpc.tp_model.tpc_minor_version == 1)

    def test_get_keras_not_supported_platform(self):
        with self.assertRaises(Exception) as e:
            mct.get_target_platform_capabilities(TENSORFLOW, "platform1")
        self.assertTrue(e.exception)

    def test_get_keras_not_supported_fw(self):
        with self.assertRaises(Exception) as e:
            mct.get_target_platform_capabilities("ONNX", DEFAULT_TP_MODEL)
        self.assertTrue(e.exception)

    def test_get_keras_not_supported_version(self):
        with self.assertRaises(Exception) as e:
            mct.get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL, "v0")
        self.assertTrue(e.exception)
