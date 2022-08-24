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
import torch

from model_compression_toolkit.core.pytorch.constants import GAMMA, BETA, MOVING_MEAN, MOVING_VARIANCE
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This test checks the BatchNorm tuning feature.
"""


def bn_weight_change(bn: torch.nn.Module):
    bw_shape = bn.weight.shape
    delattr(bn, GAMMA)
    delattr(bn, BETA)
    delattr(bn, MOVING_VARIANCE)
    delattr(bn, MOVING_MEAN)
    bn.register_buffer(GAMMA, torch.rand(bw_shape))
    bn.register_buffer(BETA, torch.rand(bw_shape))
    bn.register_buffer(MOVING_VARIANCE, torch.abs(torch.rand(bw_shape)))
    bn.register_buffer(MOVING_MEAN, torch.rand(bw_shape))
    return bn


class BNTuningNet(torch.nn.Module):
    def __init__(self):
        super(BNTuningNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn1 = bn_weight_change(self.bn1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.bn2 = bn_weight_change(self.bn2)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return x


class BNTuningNetTest(BasePytorchTest):
    """
    This test checks the BatchNorm tuning feature.
    """

    def __init__(self, unit_test, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)

    def create_feature_network(self, input_shape):
        return BNTuningNet()
