from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8 import ModelPyTorch, yaml_load

cfg_dict = yaml_load("/Vols/vol_design/tools/swat/users/ariell/repos/sony_fork/git/model_optimization/tutorials/mct_model_garden/models_pytorch/yolov8/yolov8-seg.yaml", append_filename=True)  # model dict
model = ModelPyTorch.from_pretrained("SSI-DNN/pytorch_yolov8n_inst_seg_640x640", cfg=cfg_dict, mode='segmentation')

model = model.eval()
model = model.cuda()

import model_compression_toolkit as mct
from tutorials.mct_model_garden.evaluation_metrics.coco_evaluation import coco_dataset_generator
from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8_preprocess import yolov8_preprocess_chw_transpose
from typing import Iterator, Tuple, List

REPRESENTATIVE_DATASET_FOLDER = '/data/projects/swat/datasets_src/COCO/images/val2017'
REPRESENTATIVE_DATASET_ANNOTATION_FILE = '/data/projects/swat/datasets_src/COCO/annotations/instances_val2017.json'

BATCH_SIZE = 4
n_iters = 20

# Load representative dataset
representative_dataset = coco_dataset_generator(dataset_folder=REPRESENTATIVE_DATASET_FOLDER,
                                                annotation_file=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                                                preprocess=yolov8_preprocess_chw_transpose,
                                                batch_size=BATCH_SIZE)

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
                                                        dataset_loader=representative_dataset)

# Set IMX500-v1 TPC
tpc = mct.get_target_platform_capabilities(fw_name="pytorch",
                                           target_platform_name='imx500',
                                           target_platform_version='v1')

# Specify the necessary configuration for mixed precision quantization. To keep the tutorial brief, we'll use a small set of images and omit the hessian metric for mixed precision calculations. It's important to be aware that this choice may impact the resulting accuracy.
# mp_config = mct.core.MixedPrecisionQuantizationConfig(num_of_images=5,
#                                                       use_hessian_based_scores=False)
# config = mct.core.CoreConfig(mixed_precision_config=mp_config,
config = mct.core.CoreConfig(quantization_config=mct.core.QuantizationConfig(shift_negative_activation_correction=True))

# Define target Resource Utilization for mixed precision weights quantization (75% of 'standard' 8bits quantization)
# resource_utilization_data = mct.core.pytorch_resource_utilization_data(in_model=model,
#                                                                        representative_data_gen=
#                                                                        representative_dataset_gen,
#                                                                        core_config=config,
#                                                                        target_platform_capabilities=tpc)
# resource_utilization = mct.core.ResourceUtilization(weights_memory=resource_utilization_data.weights_memory * 0.75)

# Perform post training quantization
quant_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=model,
                                                            representative_data_gen=
                                                            representative_dataset_gen,
                                                            # target_resource_utilization=resource_utilization,
                                                            core_config=config,
                                                            target_platform_capabilities=tpc)

import model_compression_toolkit as mct

mct.exporter.pytorch_export_model(model=quant_model,
                                  save_model_path='./quant_model.onnx',
                                  repr_dataset=representative_dataset_gen)

from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8 import seg_model_predict
from tutorials.mct_model_garden.evaluation_metrics.coco_evaluation import evaluate_yolov8_segmentation

# Average Precision(AP) @ [IoU = 0.50:0.95 | area = all | maxDets = 100]
# 0.375 / ir 0.375 / mc 0.388
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
#  0.655 / ir 0.655 / mc 0.711
evaluate_yolov8_segmentation(model, seg_model_predict, data_dir='/data/projects/swat/datasets_src/COCO', data_type='val2017', img_ids_limit=100, output_file='results.json', iou_thresh=0.7, conf=0.001, max_dets=300,mask_thresh=0.55)

from tutorials.mct_model_garden.evaluation_metrics.coco_evaluation import evaluate_yolov8_segmentation

# Average Precision(AP) @ [IoU = 0.50:0.95 | area = all | maxDets = 100]
# 0.342 / ir 0.342 / mc 0.352
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
#  0.630 / ir 0.629 / mc 0.699
evaluate_yolov8_segmentation(quant_model, seg_model_predict, data_dir='/data/projects/swat/datasets_src/COCO', data_type='val2017', img_ids_limit=100, output_file='results_quant.json', iou_thresh=0.7, conf=0.001, max_dets=300,mask_thresh=0.55)

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# from pycocotools.coco import COCO
# import json
# import random
#
# # Number of sets to display
# num_sets = 20
#
# # adjust results file name to view quant and gradient quant
# with open('results.json', 'r') as file:
#     results = json.load(file)
#
# # Extract unique image IDs from the results
# result_imgIds = list({result['image_id'] for result in results})
#
# dataDir = '/data/projects/swat/datasets_src/COCO'
# dataType = 'val2017'
# annFile = f'{dataDir}/annotations/instances_{dataType}.json'
# resultsFile = 'results.json'
# cocoGt = COCO(annFile)
# cocoDt = cocoGt.loadRes(resultsFile)
# plt.figure(figsize=(15, 7 * num_sets))
#
# for i in range(num_sets):
#     random_imgId = random.choice(result_imgIds)
#     img = cocoGt.loadImgs(random_imgId)[0]
#     image_path = f'{dataDir}/images/{dataType}/{img["file_name"]}'
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
#
#     plt.subplot(num_sets, 2, 2*i + 1)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title(f'Ground Truth {random_imgId}')
#
#     # Load and display ground truth annotations with bounding boxes
#     annIds = cocoGt.getAnnIds(imgIds=img['id'], iscrowd=None)
#     anns = cocoGt.loadAnns(annIds)
#     for ann in anns:
#         cocoGt.showAnns([ann], draw_bbox=True)
#         # Draw category ID on the image
#         bbox = ann['bbox']
#         plt.text(bbox[0], bbox[1], str(ann['category_id']), color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
#
#     plt.subplot(num_sets, 2, 2*i + 2)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title(f'Model Output {random_imgId}')
#
#     # Load and display model predictions with bounding boxes
#     annIdsDt = cocoDt.getAnnIds(imgIds=img['id'])
#     annsDt = cocoDt.loadAnns(annIdsDt)
#     for ann in annsDt:
#         cocoDt.showAnns([ann], draw_bbox=True)
#         # Draw category ID on the image
#         bbox = ann['bbox']
#         plt.text(bbox[0], bbox[1], str(ann['category_id']), color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
#
# plt.tight_layout()
# plt.show()
