# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

import model_compression_toolkit as mct
from tensorflow.keras.applications.mobilenet import MobileNet
import tensorflow as tf
import torch
from torch import nn
from torchvision.models import mobilenet_v2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision import transforms
from PIL import Image

"""
This tutorial demonstrates how a model (more specifically, MobileNetV1) can be
quantized and optimized using the Model Compression Toolkit (MCT). 
"""
####################################
# Install packages needed for yolov5
####################################
# seaborn
# pyyaml
# pandas

####################################
# Preprocessing images
####################################
import cv2
import numpy as np

MEAN = 127.5
STD = 127.5
RESIZE_SCALE = 256 / 224
SIZE = 224


def resize(x):
    resize_side = max(RESIZE_SCALE * SIZE / x.shape[0], RESIZE_SCALE * SIZE / x.shape[1])
    height_tag = int(np.round(resize_side * x.shape[0]))
    width_tag = int(np.round(resize_side * x.shape[1]))
    resized_img = cv2.resize(x, (width_tag, height_tag))
    offset_height = int((height_tag - SIZE) / 2)
    offset_width = int((width_tag - SIZE) / 2)
    cropped_img = resized_img[offset_height:offset_height + SIZE, offset_width:offset_width + SIZE]
    return cropped_img


def normalization(x):
    return (x - MEAN) / STD

def np_to_pil(img):
    return Image.fromarray(img)

class ConcatOut(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=0):
        super().__init__()
        self.d = dimension
    def forward(self, inputs):
        y = []
        for x in inputs:
            x = torch.sigmoid(x)
            x = torch.reshape(x, (-1,85))
            y.append(x)
        return torch.cat(y, self.d)

class Yolov5nRefactor(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.model.model # remove wrappers
        # ------------------------------------------------------ #
        # Replace Upsampe with UpsamplingBilinear2d
        # ------------------------------------------------------ #
        for block in model:
            if block._get_name() == 'Upsample':
                i, f = block.i, block.f
                model[i] = nn.UpsamplingBilinear2d(scale_factor=2.0)
                model[i].i, model[i].f = i, f

        # ------------------------------------------------------ #
        # Add concat for last convolutions in the end
        # ------------------------------------------------------ #
        last_convs_i = [17, 20, 23]
        last_convs_nn = torch.nn.Sequential()
        for i,block in enumerate(list(model)[-1].m):
            block.i = 24+i
            block.f = last_convs_i[i]
            last_convs_nn = nn.Sequential(*last_convs_nn, block)
        concat_nn = ConcatOut()
        concat_nn.f = [24,25,26]
        concat_nn.i = 27
        model = nn.Sequential(*list(model.children())[:-1]) # remove wrapper
        self.save_out_block = [4, 6, 10, 14] + last_convs_i + concat_nn.f
        self.yolov5n = nn.Sequential(*model, *last_convs_nn, concat_nn)

    def forward(self, x):
        y = []  # outputs
        for block in self.yolov5n:
            if block.f != -1:  # if not from previous layer
                x = y[block.f] if isinstance(block.f, int) else [x if j == -1 else y[j] for j in block.f]  # from earlier layers
            x = block(x)  # run block
            y.append(x if block.i in self.save_out_block else None)  # save output
        return x

if __name__ == '__main__':

    # Set the batch size of the images at each calibration iteration.
    batch_size = 10

    # Set the path to the folder of images to load and use for the representative dataset.
    # Notice that the folder have to contain at least one image.
    folder = r'E:\Datasets\representative'

    # Create a representative data generator, which returns a list of images.
    # The images can be preprocessed using a list of preprocessing functions.
    from model_compression_toolkit import FolderImageLoader
    # image_data_loader = FolderImageLoader(folder,
    #                                       preprocessing=[resize, normalization],
    #                                       batch_size=batch_size)

    image_data_loader = FolderImageLoader(folder,batch_size=batch_size,
                                          preprocessing=[np_to_pil,
                                                        transforms.Compose([transforms.Resize((640,640)),
                                                                             #transforms.CenterCrop(224),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])])

    # Create a Callable representative dataset for calibration purposes.
    # The function should be called without any arguments, and should return a list numpy arrays (array for each
    # model's input).
    # For example: A model has two input tensors - one with input shape of [32 X 32 X 3] and the second with
    # an input shape of [224 X 224 X 3]. We calibrate the model using batches of 20 images.
    # Calling representative_data_gen() should return a list
    # of two numpy.ndarray objects where the arrays' shapes are [(20, 3, 32, 32), (20, 3, 224, 224)].
    def representative_data_gen() -> list:
        return [image_data_loader.sample()]

    # Create a model and quantize it using the representative_data_gen as the calibration images.
    # Set the number of calibration iterations to 10.
    #model = tf.keras.models.load_model(model_path)
    #model = tf.saved_model.load(model_path)
    #model = mobilenet_v2(pretrained=True)
    #model = ssdlite320_mobilenet_v3_large(pretrained=True)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', autoshape=False, pretrained=True)
    model = Yolov5nRefactor(model)

    # x = torch.randn((1,3,640,640))
    # y = model(x)

    quantized_model, quantization_info = mct.pytorch_post_training_quantization(model, representative_data_gen, n_iter=10)

    print("Done!")