import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from tutorials.mct_model_garden.models_pytorch.yolov8.coco_dataset import get_transform, coco_evaluate, pre_process
from tutorials.quick_start.pytorch_fw.ultralytics_lib.common_replacers import YOLOReplacer
from yolov8 import yolov8_pytorch

REPRESENTATIVE_DATASET_FOLDER = '/data/projects/swat/datasets_src/COCO/images/val2017'
REPRESENTATIVE_DATASET_ANNOTATION_FILE = '/data/projects/swat/datasets_src/COCO/annotations/instances_val2017.json'


dataset = CocoDetection(root=REPRESENTATIVE_DATASET_FOLDER, annFile=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                        transform=get_transform())
# evaluator = CocoEvaluator(coco_gt=dataset.coco, iou_types=["bbox"])
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size)

output_resize = (1, 1)
# img = torch.randn(*[batch_size, 3, 640, 640], device='cuda')
img = torch.zeros(*[batch_size, 3, 640, 640], device='cuda')
model = yolov8_pytorch("yolov8n.yaml").to(device='cuda')
# model(img)
org_dict = torch.load("/Vols/vol_design/tools/swat/users/ariell/repos/my_fork/yolov8_tut/yolov8n.pt")
org_model = org_dict['model'].to(device='cuda')
model.load_state_dict(org_model.state_dict())
model = model.eval()
org_out = org_model(img.type(torch.cuda.HalfTensor))
out = model(img)
rand_img = torch.rand(*[batch_size, 3, 640, 640], device='cuda')
org_rand = org_model(rand_img.type(torch.cuda.HalfTensor))
out_rand = model(rand_img)
print('model(img)')
INPUT_RESOLUTION = 640
output_resize = {'shape': (INPUT_RESOLUTION, INPUT_RESOLUTION), 'aspect_ratio_preservation': True}
eval_results = coco_evaluate(model=model,
                             dataset_folder=REPRESENTATIVE_DATASET_FOLDER,
                             annotation_file=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                             preprocess=pre_process,
                             output_resize=output_resize,
                             batch_size=batch_size,
                             original=False)
print(eval_results)
eval_results = coco_evaluate(model=org_model,
                             dataset_folder=REPRESENTATIVE_DATASET_FOLDER,
                             annotation_file=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                             preprocess=pre_process,
                             output_resize=output_resize,
                             batch_size=batch_size,
                             original=True)
print(eval_results)

# for img, label in dataloader:
#     predictions = model(img.to('cuda'))
#     org_predictions = org_model(img.type(torch.cuda.HalfTensor))
#     delta = org_predictions[0] - predictions[0]
#     print(delta.max())
#     YOLOReplacer
#     # evaluator.update(predictions)
#
# # evaluator.synchronize_between_processes()
# # evaluator.accumulate()
# # evaluator.summarize()
#
# # load weights
# # eval
# # mct
# # eval
