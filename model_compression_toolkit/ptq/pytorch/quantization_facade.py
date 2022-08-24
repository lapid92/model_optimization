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
from typing import Callable

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import PYTORCH, FOUND_TORCH
from model_compression_toolkit.core.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit import CoreConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.runner import core_runner, _init_tensorboard_writer
from model_compression_toolkit.ptq.runner import ptq_runner
from model_compression_toolkit.core.exporter import export_model
from model_compression_toolkit.core.analyzer import analyzer_model_quantization


if FOUND_TORCH:
    from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
    from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
    from model_compression_toolkit.core.pytorch.constants import DEFAULT_TP_MODEL
    from torch.nn import Module
    import copy

    import numpy as np
    from tqdm import tqdm
    import torch
    import matplotlib.pyplot as plt
    from model_compression_toolkit.core.pytorch.utils import set_model, to_torch_tensor

    from model_compression_toolkit import get_target_platform_capabilities
    DEFAULT_PYTORCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)


    def _apply_bn_tuning(quantized_model, core_config, representative_data_gen, tb_w):
        model = copy.deepcopy(quantized_model)
        set_model(model)

        # User control for now?
        # apply fine tune for running mean?
        bn_tuning_mean = True
        # apply fine tune for running var?
        bn_tuning_var = True

        show_graphs = False

        bn_num = 0

        # Move every BN to train mode and count bn in model
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()
                bn_num = bn_num + 1

        first_mean = [] * bn_num
        first_var = [] * bn_num

        last_mean = [] * bn_num
        last_var = [] * bn_num
        if show_graphs:
            bn_mean_from_orig_array = np.zeros([bn_num, core_config.n_iter])
            bn_var_from_orig_array = np.zeros([bn_num, core_config.n_iter])

            bn_mean_array = np.zeros([bn_num, core_config.n_iter])
            bn_var_array = np.zeros([bn_num, core_config.n_iter])

        momentum = 0

        bn_names = [] * bn_num

        # get orig values of running mean and running var
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_names.append(name)
                momentum = module.momentum
                first_mean.append(copy.deepcopy(module.running_mean))
                first_var.append(copy.deepcopy(module.running_var))
                last_mean.append(copy.deepcopy(module.running_mean))
                last_var.append(copy.deepcopy(module.running_var))

        for iter in tqdm(range(core_config.n_iter)):
            with torch.no_grad():
                model(to_torch_tensor(representative_data_gen()[0]))
                i = 0
                for module in model.modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        if not bn_tuning_mean:
                            module.running_mean = copy.deepcopy(first_mean[i])
                        if not bn_tuning_var:
                            module.running_var = copy.deepcopy(first_var[i])

                        if show_graphs:
                            bn_mean_from_orig_array[i, iter] = torch.norm(module.running_mean - first_mean[i], p=2) \
                                                               / torch.norm(first_mean[i], p=2)
                            bn_var_from_orig_array[i, iter] = torch.norm(module.running_var - first_var[i], p=2) \
                                                              / torch.norm(first_var[i], p=2)

                            bn_mean_array[i, iter] = torch.norm(module.running_mean - last_mean[i], p=2) \
                                                     / torch.norm(first_mean[i], p=2)
                            bn_var_array[i, iter] = torch.norm(module.running_var - last_var[i], p=2) \
                                                    / torch.norm(first_var[i], p=2)
                            last_mean[i] = copy.deepcopy(module.running_mean)
                            last_var[i] = copy.deepcopy(module.running_var)
                        i = i + 1
        set_model(model)

        if show_graphs:
            for bn in range(bn_num):
                plt.figure()
                plt.plot(bn_mean_array[bn], 'b-', label='mean')
                plt.plot(bn_var_array[bn], 'g-', label='var')
                plt.title(bn_names[bn] + ', momentum=' + str(momentum))
                plt.xlabel('Iters')
                plt.ylabel('Norm2 of diff between iters')
                plt.legend(loc="upper right")
                plt.show()

            for bn in range(bn_num):
                plt.figure()
                plt.plot(bn_mean_from_orig_array[bn], 'b-', label='mean')
                plt.plot(bn_var_from_orig_array[bn], 'g-', label='var')
                plt.title(bn_names[bn] + ', diff from original, momentum=' + str(momentum))
                plt.xlabel('Iters')
                plt.ylabel('Norm2 of diff from orig')
                plt.legend(loc="upper right")
                plt.show()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        return model

    def pytorch_post_training_quantization_experimental(in_module: Module,
                                                        representative_data_gen: Callable,
                                                        target_kpi: KPI = None,
                                                        core_config: CoreConfig = CoreConfig(),
                                                        fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                                                        target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_PYTORCH_TPC):
        """
        Quantize a trained Pytorch module using post-training quantization.
        By default, the module is quantized using a symmetric constraint quantization thresholds
        (power of two) as defined in the default TargetPlatformCapabilities.
        The module is first optimized using several transformations (e.g. BatchNormalization folding to
        preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
        being collected for each layer's output (and input, depends on the quantization configuration).
        Thresholds are then being calculated using the collected statistics and the module is quantized
        (both coefficients and activations by default).
        If gptq_config is passed, the quantized weights are optimized using gradient based post
        training quantization by comparing points between the float and quantized modules, and minimizing the
        observed loss.

        Args:
            in_module (Module): Pytorch module to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.
            core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
            fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default PyTorch info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/pytorch/default_framework_info.py>`_
            target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the PyTorch model according to. `Default PyTorch TPC <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/tpc_models/pytorch_tp_models/pytorch_default.py>`_


        Returns:
            A quantized module and information the user may need to handle the quantized module.

        Examples:

            Import a Pytorch module:

            >>> import torchvision.models.mobilenet_v2 as models
            >>> module = models.mobilenet_v2()

            Create a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

            Import mct and pass the module with the representative dataset generator to get a quantized module:

            >>> import model_compression_toolkit as mct
            >>> quantized_module, quantization_info = mct.pytorch_post_training_quantization(module, repr_datagen)

        """

        if core_config.mixed_precision_enable:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfigV2):
                common.Logger.error("Given quantization config to mixed-precision facade is not of type "
                                    "MixedPrecisionQuantizationConfigV2. Please use pytorch_post_training_quantization API,"
                                    "or pass a valid mixed precision configuration.")

            common.Logger.info("Using experimental mixed-precision quantization. "
                               "If you encounter an issue please file a bug.")

        tb_w = _init_tensorboard_writer(fw_info)

        fw_impl = PytorchImplementation()

        tg, bit_widths_config = core_runner(in_model=in_module,
                                            representative_data_gen=representative_data_gen,
                                            core_config=core_config,
                                            fw_info=fw_info,
                                            fw_impl=fw_impl,
                                            tpc=target_platform_capabilities,
                                            target_kpi=target_kpi,
                                            tb_w=tb_w)

        tg = ptq_runner(tg, fw_info, fw_impl, tb_w)

        if core_config.debug_config.analyze_similarity:
            analyzer_model_quantization(representative_data_gen, tb_w, tg, fw_impl, fw_info)

        quantized_model, user_info = export_model(tg, fw_info, fw_impl, tb_w, bit_widths_config)
        set_model(quantized_model)
        if core_config.quantization_config.bn_tuning:
            quantized_model = _apply_bn_tuning(quantized_model, core_config, representative_data_gen, tb_w)

        return quantized_model, user_info

else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def pytorch_post_training_quantization_experimental(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_post_training_quantization_experimental. '
                        'Could not find the torch package.')
