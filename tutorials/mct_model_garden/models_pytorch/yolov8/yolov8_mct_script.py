from typing import Iterator, Tuple, List

import numpy as np
import torch

import model_compression_toolkit as mct
from tutorials.mct_model_garden.models_pytorch.yolov8.coco_dataset import pre_process, \
    CoCoDataset, coco_evaluate
from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8_script import INPUT_RESOLUTION
from yolov8 import yolov8_pytorch

REPRESENTATIVE_DATASET_FOLDER = '/data/projects/swat/datasets_src/COCO/images/val2017'
REPRESENTATIVE_DATASET_ANNOTATION_FILE = '/data/projects/swat/datasets_src/COCO/annotations/instances_val2017.json'

model = yolov8_pytorch("yolov8n.yaml").to(device='cuda')
org_dict = torch.load("/Vols/vol_design/tools/swat/users/ariell/repos/my_fork/yolov8_tut/yolov8n.pt")
org_model = org_dict['model'].to(device='cuda')
model.load_state_dict(org_model.state_dict())
model = model.eval()

n_iters = 20
batch_size = 4

# Set IMX500-v1 TPC
tpc = mct.get_target_platform_capabilities("pytorch", 'imx500', target_platform_version='v1')


def random_datagen():
    yield [np.random.random((batch_size, 3, 640, 640))]


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
            yield [next(ds_iter)[1]]

    return representative_dataset


coco_dataset = CoCoDataset(dataset_path=REPRESENTATIVE_DATASET_FOLDER,
                           _gt_annotations_path=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                           preprocessing=pre_process)

# Perform post training quantization
quant_model, _ = mct.ptq.pytorch_post_training_quantization(model,
                                                            get_representative_dataset(n_iters,
                                                                                       coco_dataset.data_generator(
                                                                                           batch_size)),
                                                            target_platform_capabilities=tpc)
print('ptq is ready')
output_resize = {'shape': (INPUT_RESOLUTION, INPUT_RESOLUTION), 'aspect_ratio_preservation': True}
quant_results = coco_evaluate(model=quant_model,
                              dataset_folder=REPRESENTATIVE_DATASET_FOLDER,
                              annotation_file=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                              preprocess=pre_process,
                              output_resize=output_resize,
                              batch_size=batch_size,
                              original=False)
print(quant_results)
