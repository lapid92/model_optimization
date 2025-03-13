import torch

from model_compression_toolkit.core import BitWidthConfig
from model_compression_toolkit.core.common.network_editors import NodeNameScopeFilter, NodeNameFilter
from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8 import ModelPyTorch, yaml_load, KeyPointsPostProcessWrapper
from tutorials.notebooks.imx500_notebooks.pytorch.yolo8n_helper_funcs_coco_evaluation import coco_evaluate

# Load the adjusted model from hugging-face
cfg_dict = yaml_load(
    "/Vols/vol_design/tools/swat/users/ariell/repos/sony_fork/git/model_optimization/tutorials/mct_model_garden/models_pytorch/yolov8/yolov8n-pose.yaml",
    append_filename=True)

model = ModelPyTorch(cfg=cfg_dict)

PRETRAINED_WEIGHTS_FILE = '/Vols/vol_design/tools/swat/users/idanb/repository/git4/untracked/yolov8n-pose.pt'
pretrained_weights = torch.load(PRETRAINED_WEIGHTS_FILE)['model'].state_dict()
model.load_state_dict(pretrained_weights, strict=False)


# Ensure the model is in evaluation mode
model = model.eval()

import model_compression_toolkit as mct
from tutorials.mct_model_garden.evaluation_metrics.coco_evaluation import CocoDataset, DataLoader
from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8_preprocess import yolov8_preprocess_chw_transpose
from typing import Iterator, Tuple, List

REPRESENTATIVE_DATASET_FOLDER = '/data/projects/swat/datasets_src/COCO/images/val2017'
REPRESENTATIVE_DATASET_ANNOTATION_FILE = '/data/projects/swat/datasets_src/COCO/annotations/person_keypoints_val2017.json'
BATCH_SIZE = 4
n_iters = 20

representative_dataset = CocoDataset(dataset_folder=REPRESENTATIVE_DATASET_FOLDER,
                                     annotation_file=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                                     preprocess=yolov8_preprocess_chw_transpose)

representative_dataloader = DataLoader(representative_dataset, BATCH_SIZE, shuffle=True)


# Define representative dataset generator
def get_representative_dataset(n_iter: int, dataset_loader: Iterator[Tuple]):
    """
    This function creates a representative dataset generator. The generator yields numpy
        arrays of batches of shape: [Batch, H, W ,C].
    Args:
        n_iter: number of iterations for MCT to calibrate on
    Returns:
        A representative dataset generator
    """

    def representative_dataset() -> Iterator[List]:
        ds_iter = iter(dataset_loader)
        for _ in range(n_iter):
            yield [next(ds_iter)[0]]

    return representative_dataset


# Get representative dataset generator
representative_dataset_gen = get_representative_dataset(n_iter=n_iters,
                                                        dataset_loader=representative_dataloader)

# # Set IMX500-v1 TPC
# tpc = mct.get_target_platform_capabilities(fw_name="pytorch",
#                                            target_platform_name='imx500',
#                                            target_platform_version='v1')
# Set IMX500-v3 TPC
tpc = mct.get_target_platform_capabilities(fw_name="pytorch",
                                           target_platform_name='imx500',
                                           target_platform_version='v3')
# Configure MCT manually for specific layers
manual_bit_cfg = BitWidthConfig()
# 43.36552807286669

# 0.4121
manual_bit_cfg.set_manual_activation_bit_width(
    [NodeNameScopeFilter('mul'),
     NodeNameScopeFilter('sub'),
     NodeNameScopeFilter('sub_1'),
     NodeNameScopeFilter('add_6'),
     NodeNameScopeFilter('add_7'),
     NodeNameScopeFilter('stack'),
     NodeNameScopeFilter('mul_1'),
     NodeNameScopeFilter('mul_2'),
     NodeNameScopeFilter('mul_3'),
     NodeNameScopeFilter('mul_4'),
     NodeNameScopeFilter('add_8'),
     NodeNameScopeFilter('add_9'),
     ], 16)

# Specify the necessary configuration for mixed precision quantization. To keep the tutorial brief, we'll use a small set of images and omit the hessian metric for mixed precision calculations. It's important to be aware that this choice may impact the resulting accuracy.
mp_config = mct.core.MixedPrecisionQuantizationConfig(num_of_images=10)

# Specify the necessary configuration for mixed precision quantization
config = mct.core.CoreConfig(mixed_precision_config=mp_config,
                             quantization_config=mct.core.QuantizationConfig(shift_negative_activation_correction=True,
                                                                             concat_threshold_update=True),
                             bit_width_config=manual_bit_cfg)

# Define target Resource Utilization for mixed precision weights quantization (80% of 'standard' 8bits quantization)
resource_utilization_data = mct.core.pytorch_resource_utilization_data(in_model=model,
                                                                       representative_data_gen=representative_dataset_gen,
                                                                       core_config=config,
                                                                       target_platform_capabilities=tpc)
mp_per = 73
save_model_path = './qmodel_' + str(mp_per) + '.onnx'
save_model_path_pp = './qmodel_pp_' + str(mp_per) + '.onnx'

resource_utilization = mct.core.ResourceUtilization(weights_memory=resource_utilization_data.weights_memory * mp_per/100)

# Perform post training quantization
quant_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=model,
                                                            representative_data_gen=representative_dataset_gen,
                                                            target_resource_utilization=resource_utilization,
                                                            core_config=config,
                                                            target_platform_capabilities=tpc)
print('Quantized model is ready')

# Wrapped the model with PostProcess NMS.
# Define PostProcess params
score_threshold = 0.001
iou_threshold = 0.7
max_detections = 300

quant_model_pp = KeyPointsPostProcessWrapper(model=quant_model,
                                             score_threshold=score_threshold,
                                             iou_threshold=iou_threshold,
                                             max_detections=max_detections)

model_pp = KeyPointsPostProcessWrapper(model=model,
                                       score_threshold=score_threshold,
                                       iou_threshold=iou_threshold,
                                       max_detections=max_detections)

mct.exporter.pytorch_export_model(model=quant_model,
                                  save_model_path=save_model_path,
                                  repr_dataset=representative_dataset_gen)

mct.exporter.pytorch_export_model(model=quant_model_pp,
                                  save_model_path=save_model_path_pp,
                                  repr_dataset=representative_dataset_gen)

from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8 import keypoints_model_predict
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device

EVAL_DATASET_FOLDER = '/data/projects/swat/datasets_src/COCO/images/val2017'
EVAL_DATASET_ANNOTATION_FILE = '/data/projects/swat/datasets_src/COCO/annotations/person_keypoints_val2017.json'
INPUT_RESOLUTION = 640

# Define resizing information to map between the model's output and the original image dimensions
output_resize = {'shape': (INPUT_RESOLUTION, INPUT_RESOLUTION), 'aspect_ratio_preservation': True,
                 'normalized_coords': False}

# Evaluate the quantized model with PostProcess on coco
eval_results = coco_evaluate(model=quant_model_pp.to(get_working_device()),
                             dataset_folder=EVAL_DATASET_FOLDER,
                             annotation_file=EVAL_DATASET_ANNOTATION_FILE,
                             preprocess=yolov8_preprocess_chw_transpose,
                             output_resize=output_resize,
                             batch_size=BATCH_SIZE,
                             model_inference=keypoints_model_predict,
                             task='Keypoints')

print("Quantized model mAP: {:.4f}".format(eval_results[0]))
