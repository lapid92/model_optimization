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

import torch
from torch.nn import Linear, LayerNorm, ReLU

import model_compression_toolkit as mct
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.exporter import pytorch_export_model, PytorchExportSerializationFormat, \
    QuantizationFormat
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, WEIGHTS_N_BITS
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness, \
    AttributeQuantizationConfig

kernel_weights_n_bits = 8
bias_weights_n_bits = 32
activation_n_bits = 8


def get_op_qco(kernel_n_bits, bias_n_bits):
    # define a default quantization config for all non-specified weights attributes.
    default_weight_attr_config = AttributeQuantizationConfig()

    # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
    kernel_base_config = AttributeQuantizationConfig(
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True)

    base_cfg = schema.OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)

    default_config = schema.OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO
    )

    mx_cfg_list = [base_cfg]
    if kernel_weights_n_bits is not None:
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: kernel_n_bits}}))
    if bias_weights_n_bits is not None:
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: bias_n_bits}}))

    return base_cfg, mx_cfg_list, default_config


def generate_tpc_local(default_config, base_config, mixed_precision_cfg_list):
    default_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple([default_config]))
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple(mixed_precision_cfg_list),
        base_config=base_config)

    operator_set = []

    conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
    relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU)
    add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD)
    operator_set.extend([conv, relu, add])

    generated_tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        operator_set=tuple(operator_set))

    return generated_tpc

def representative_data_gen(shape=(3, 8, 10), num_inputs=1, batch_size=2, num_iter=1):
    for _ in range(num_iter):
        yield [torch.randn(batch_size, *shape)] * num_inputs

class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Linear(10, 12)
        self.norm1 = LayerNorm(12)
        self.mlp_fc1 = Linear(12, 12)
        self.relu = ReLU()
        self.mlp_fc2 = Linear(12, 12)
        self.norm_head = LayerNorm(12)
        self.head = Linear(12, 12)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.mlp_fc2(self.relu(self.mlp_fc1(self.norm1(x))))
        x = self.norm_head(x)
        return self.head(x)
class BaseModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Linear(10, 12)
        self.norm1 = LayerNorm(12)
        self.mlp_fc1 = Linear(12, 12)
        self.relu = ReLU()
        self.mlp_fc2 = Linear(12, 12)
        self.norm2 = LayerNorm(12)
        self.mlp_fc3 = Linear(12, 12)
        self.relu2 = ReLU()
        self.mlp_fc4 = Linear(12, 12)
        self.dummy_norm = LayerNorm(10)
        self.dummy_mlp_fc = Linear(10, 12)
        self.dummy_relu = ReLU()
        self.dummy_mlp_fc2 = Linear(12, 12)
        self.norm_head = LayerNorm(12)
        self.head = Linear(12, 12)

    def forward(self, x):
        y = self.dummy_mlp_fc2(self.dummy_relu(self.dummy_mlp_fc(self.dummy_norm(x))))
        x = self.embed(x)
        y = x + y
        x = self.mlp_fc2(self.relu(self.mlp_fc1(self.norm1(x))))
        y = y + self.mlp_fc4(self.relu2(self.mlp_fc3(self.norm2(y))))
        y = self.norm_head(y)
        head = self.head(y)
        return x + head


class BaseModel3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Linear(10, 12)
        self.norm1 = LayerNorm(12)
        self.mlp_fc1 = Linear(12, 12)
        self.relu = ReLU()
        self.mlp_fc2 = Linear(12, 12)
        self.norm2 = LayerNorm(10)
        self.mlp_fc3 = Linear(10, 12)
        self.relu2 = ReLU()
        self.mlp_fc4 = Linear(12, 12)
        self.dummy_norm = LayerNorm(10)
        self.dummy_mlp_fc = Linear(10, 12)
        self.dummy_relu = ReLU()
        self.dummy_mlp_fc2 = Linear(12, 12)
        self.norm_head = LayerNorm(12)
        self.head = Linear(12, 12)

    def forward(self, x):
        x1 = self.embed(x)
        x1_1 = self.mlp_fc2(self.relu(self.mlp_fc1(self.norm1(x1))))
        x2 = self.mlp_fc4(self.relu2(self.mlp_fc3(self.norm2(x))))
        y = x1 + x2
        head = self.head(y)
        return x1_1 + head


class TestRotation:

    def get_tpc(self, kernel_n_bits, bias_n_bits):
        base_cfg, mx_cfg_list, default_config = get_op_qco(kernel_n_bits, bias_n_bits)
        tpc = generate_tpc_local(default_config, base_cfg, mx_cfg_list)
        return tpc

    def test_rotation_1(self):
        float_model = BaseModel()

        target_platform_cap = self.get_tpc(kernel_n_bits=8, bias_n_bits=8)

        core_config = CoreConfig()
        core_config.enable_rotation = True

        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=float_model,
            representative_data_gen=representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=target_platform_cap
        )
        print(quantized_model)
        pytorch_export_model(quantized_model,
                             save_model_path='aa.onnx',
                             repr_dataset=representative_data_gen,
                             serialization_format=PytorchExportSerializationFormat.ONNX,
                             quantization_format=QuantizationFormat.MCTQ)


    def test_rotation_2(self):
        float_model = BaseModel2()

        target_platform_cap = self.get_tpc(kernel_n_bits=8, bias_n_bits=8)

        core_config = CoreConfig()
        core_config.enable_rotation = True

        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=float_model,
            representative_data_gen=representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=target_platform_cap
        )
        print(quantized_model)
        pytorch_export_model(quantized_model,
                             save_model_path='aa.onnx',
                             repr_dataset=representative_data_gen,
                             serialization_format=PytorchExportSerializationFormat.ONNX,
                             quantization_format=QuantizationFormat.MCTQ)


    def test_rotation_3(self):
        float_model = BaseModel3()

        target_platform_cap = self.get_tpc(kernel_n_bits=8, bias_n_bits=8)

        core_config = CoreConfig()
        core_config.enable_rotation = True

        quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=float_model,
            representative_data_gen=representative_data_gen,
            core_config=core_config,
            target_platform_capabilities=target_platform_cap
        )
        print(quantized_model)
        pytorch_export_model(quantized_model,
                             save_model_path='aa.onnx',
                             repr_dataset=representative_data_gen,
                             serialization_format=PytorchExportSerializationFormat.ONNX,
                             quantization_format=QuantizationFormat.MCTQ)
