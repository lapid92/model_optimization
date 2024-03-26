from PIL import Image
import json
import os
from typing import Callable, Any, Tuple

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

from .post_process_google_em import PostProcessGoogleMeanAveragePrecision

REPRESENTATIVE_DATASET_FOLDER = '/data/projects/swat/datasets_src/COCO/images/val2017'
REPRESENTATIVE_DATASET_ANNOTATION_FILE = '/data/projects/swat/datasets_src/COCO/annotations/instances_val2017.json'


class CoCoDataset:
    def __init__(self, dataset_path, _gt_annotations_path, preprocessing, background: bool = False,
                 research_dir: str = None):
        self._imgs_path = dataset_path
        self._gt_annotations_path = _gt_annotations_path
        from pycocotools.coco import COCO
        self._coco = COCO(self._gt_annotations_path)
        # self.set_evaluation_metric(em_type)
        self._sorted_ids, self._id_to_ann = self._build_id_to_annotation_dict()
        self._id_to_index = self._build_id_to_index()
        self._background = background
        self.preprocessing = preprocessing

    def _preprocess(self, img, annotations):
        for preprocess in self.preprocessing:
            img, annotations = preprocess(img, annotations)
        return img, annotations

    def _build_bbox_id_to_annotation_dict(self):
        all_imgs_ids = sorted(self._coco.imgs.keys())
        id_to_ann = {}
        for id in all_imgs_ids:
            ann = self._coco.imgToAnns[id]
            if ann is not None and ann != []:
                id_to_ann[id] = ann
        sorted_ids = sorted(id_to_ann.keys())
        return sorted_ids, id_to_ann

    def _build_id_to_annotation_dict(self):
        return self._build_bbox_id_to_annotation_dict()

    def _get_object_detection_sample(self, index):
        img_data = self._coco.imgs[self._sorted_ids[index]]
        img_id = img_data['id']
        img = np.uint8(np.array(Image.open(os.path.join(self._imgs_path, img_data['file_name'])).convert('RGB')))
        img_annotation = self._id_to_ann[img_id]
        boxes = np.asarray([ia['bbox'] for ia in img_annotation])
        classes = np.asarray([ia['category_id'] for ia in img_annotation])
        label = {'image_id': int(img_id), 'boxes': boxes, 'classes': classes,
                 'orig_img_dims': np.shape(img)}
        return str(img_id), img, label

    def _get_validation_sample_by_index(self, index):
        return self._get_object_detection_sample(index)

    def get_sample_by_index(self, index: int):
        return self._get_validation_sample_by_index(index)

    def get_image_path_by_index(self, index: int):
        img_file_name = self._coco.imgs[self._sorted_ids[index]]['file_name']
        return os.path.join(self._imgs_path, img_file_name)

    def _build_id_to_index(self):
        id_to_index = {}
        for index in range(len(self._id_to_ann)):
            img_data = self._coco.imgs[self._sorted_ids[index]]
            img_id = str(img_data['id'])
            id_to_index[img_id] = index
        return id_to_index

    def get_original_image_by_id(self, id):
        index = self._id_to_index[id]
        return self._get_validation_sample_by_index(index)

    def get_dataset_size(self):
        return len(self._id_to_ann)

    def handle_background(self, with_background: bool, labels):
        if with_background:
            return labels
        else:
            if isinstance(labels, list):
                for label in labels:
                    label['classes'] -= 1
            else:
                labels['classes'] -= 1
            return labels

    def data_generator(self, batch_size: int, start=0, preprocessed=True, dtype=np.float32):
        data_names, data, labels = [], [], []
        for i in range(start, self.get_dataset_size()):
            n, d, l = self.get_sample_by_index(i)
            if preprocessed:
                d, l = self._preprocess(d, l)
            d = d.astype(dtype)
            data_names.append(n)
            data.append(torch.from_numpy(d))
            labels.append(l)
            if len(data) == batch_size:
                img = data
                annotations = self.handle_background(self._background, labels)
                yield data_names, torch.stack(img), annotations
                # yield data_names, np.array(data), np.array(labels)
                data_names, data, labels = [], [], []
        # if len(data) > 0:
        #     img = data
        #     annotations = self.handle_background(self._background, labels)
        #     yield data_names, img, annotations
        #     # yield np.array(data_names), np.array(data), np.array(labels)
        # img_id = data_names
        # img = np.array(data)
        # annotations = np.array(labels)
        # if preprocessed:
        #     img, annotations = self._preprocess(img, annotations)
        # annotations = self.handle_background(self._background, annotations)
        # img = img.astype(dtype)
        # return img_id, img, annotations


#
# class BoxFormat(Enum):
#     YMIM_XMIN_YMAX_XMAX = 'ymin_xmin_ymax_xmax'
#     XMIM_YMIN_XMAX_YMAX = 'xmin_ymin_xmax_ymax'
#     XMIN_YMIN_W_H = 'xmin_ymin_width_height'
#     XC_YC_W_H = 'xc_yc_width_height'
#
#
# class SortOrder(object):
#     """Enum class for sort order.
#
#   Attributes:
#     ascend: ascend order.
#     descend: descend order.
#   """
#     ASCEND = 1
#     DESCEND = 2
#
#
# class BoxList(object):
#     """Box collection.
#
#   BoxList represents a list of bounding boxes as numpy array, where each
#   bounding box is represented as a row of 4 numbers,
#   [y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes within a
#   given list correspond to a single image.
#
#   Optionally, users can add additional related fields (such as
#   objectness/classification scores).
#   """
#
#     def __init__(self, data):
#         """Constructs box collection.
#
#     Args:
#       data: a numpy array of shape [N, 4] representing box coordinates
#
#     Raises:
#       ValueError: if bbox data is not a numpy array
#       ValueError: if invalid dimensions for bbox data
#     """
#         if not isinstance(data, np.ndarray):
#             raise ValueError('data must be a numpy array.')
#         if len(data.shape) != 2 or data.shape[1] != 4:
#             raise ValueError('Invalid dimensions for box data.')
#         if data.dtype != np.float32 and data.dtype != np.float64:
#             raise ValueError('Invalid data type for box data: float is required.')
#         if not self._is_valid_boxes(data):
#             raise ValueError('Invalid box data. data must be a numpy array of '
#                              'N*[y_min, x_min, y_max, x_max]')
#         self.data = {'boxes': data}
#
#     def num_boxes(self):
#         """Return number of boxes held in collections."""
#         return self.data['boxes'].shape[0]
#
#     def get_extra_fields(self):
#         """Return all non-box fields."""
#         return [k for k in self.data.keys() if k != 'boxes']
#
#     def has_field(self, field):
#         return field in self.data
#
#     def add_field(self, field, field_data):
#         """Add data to a specified field.
#
#     Args:
#       field: a string parameter used to speficy a related field to be accessed.
#       field_data: a numpy array of [N, ...] representing the data associated
#           with the field.
#     Raises:
#       ValueError: if the field is already exist or the dimension of the field
#           data does not matches the number of boxes.
#     """
#         if self.has_field(field):
#             raise ValueError('Field ' + field + 'already exists')
#         if len(field_data.shape) < 1 or field_data.shape[0] != self.num_boxes():
#             raise ValueError('Invalid dimensions for field data')
#         self.data[field] = field_data
#
#     def get(self):
#         """Convenience function for accesssing box coordinates.
#
#     Returns:
#       a numpy array of shape [N, 4] representing box corners
#     """
#         return self.get_field('boxes')
#
#     def get_field(self, field):
#         """Accesses data associated with the specified field in the box collection.
#
#     Args:
#       field: a string parameter used to speficy a related field to be accessed.
#
#     Returns:
#       a numpy 1-d array representing data of an associated field
#
#     Raises:
#       ValueError: if invalid field
#     """
#         if not self.has_field(field):
#             raise ValueError('field {} does not exist'.format(field))
#         return self.data[field]
#
#     def get_coordinates(self):
#         """Get corner coordinates of boxes.
#
#     Returns:
#      a list of 4 1-d numpy arrays [y_min, x_min, y_max, x_max]
#     """
#         box_coordinates = self.get()
#         y_min = box_coordinates[:, 0]
#         x_min = box_coordinates[:, 1]
#         y_max = box_coordinates[:, 2]
#         x_max = box_coordinates[:, 3]
#         return [y_min, x_min, y_max, x_max]
#
#     def _is_valid_boxes(self, data):
#         """Check whether data fullfills the format of N*[ymin, xmin, ymax, xmin].
#
#     Args:
#       data: a numpy array of shape [N, 4] representing box coordinates
#
#     Returns:
#       a boolean indicating whether all ymax of boxes are equal or greater than
#           ymin, and all xmax of boxes are equal or greater than xmin.
#     """
#         if data.shape[0] > 0:
#             for i in range(data.shape[0]):
#                 if data[i, 0] > data[i, 2] or data[i, 1] > data[i, 3]:
#                     return False
#         return True
#
#
# def iou(box_mask_list1, box_mask_list2):
#     """Computes pairwise intersection-over-union between box and mask collections.
#
#   Args:
#     box_mask_list1: BoxMaskList holding N boxes and masks
#     box_mask_list2: BoxMaskList holding M boxes and masks
#
#   Returns:
#     a numpy array with shape [N, M] representing pairwise iou scores.
#   """
#     return np_mask_ops_iou(box_mask_list1.get_masks(),
#                            box_mask_list2.get_masks())
#
#
# def np_mask_ops_iou(masks1, masks2):
#     """Computes pairwise intersection-over-union between mask collections.
#
#   Args:
#     masks1: a numpy array with shape [N, height, width] holding N masks. Masks
#       values are of type np.uint8 and values are in {0,1}.
#     masks2: a numpy array with shape [M, height, width] holding N masks. Masks
#       values are of type np.uint8 and values are in {0,1}.
#
#   Returns:
#     a numpy array with shape [N, M] representing pairwise iou scores.
#
#   Raises:
#     ValueError: If masks1 and masks2 are not of type np.uint8.
#   """
#     if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
#         raise ValueError('masks1 and masks2 should be of type np.uint8')
#     intersect = intersection(masks1, masks2)
#     area1 = area(masks1)
#     area2 = area(masks2)
#     union = np.expand_dims(area1, axis=1) + np.expand_dims(
#         area2, axis=0) - intersect
#     return intersect / np.maximum(union, EPSILON)
#
#
# EPSILON = 1e-7
#
#
# def area(masks):
#     """Computes area of masks.
#
#   Args:
#     masks: Numpy array with shape [N, height, width] holding N masks. Masks
#       values are of type np.uint8 and values are in {0,1}.
#
#   Returns:
#     a numpy array with shape [N*1] representing mask areas.
#
#   Raises:
#     ValueError: If masks.dtype is not np.uint8
#   """
#     if masks.dtype != np.uint8:
#         raise ValueError('Masks type should be np.uint8')
#     return np.sum(masks, axis=(1, 2), dtype=np.float32)
#
#
# def intersection(masks1, masks2):
#     """Compute pairwise intersection areas between masks.
#
#   Args:
#     masks1: a numpy array with shape [N, height, width] holding N masks. Masks
#       values are of type np.uint8 and values are in {0,1}.
#     masks2: a numpy array with shape [M, height, width] holding M masks. Masks
#       values are of type np.uint8 and values are in {0,1}.
#
#   Returns:
#     a numpy array with shape [N*M] representing pairwise intersection area.
#
#   Raises:
#     ValueError: If masks1 and masks2 are not of type np.uint8.
#   """
#     if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
#         raise ValueError('masks1 and masks2 should be of type np.uint8')
#     n = masks1.shape[0]
#     m = masks2.shape[0]
#     answer = np.zeros([n, m], dtype=np.float32)
#     for i in np.arange(n):
#         for j in np.arange(m):
#             answer[i, j] = np.sum(np.minimum(masks1[i], masks2[j]), dtype=np.float32)
#     return answer
#
#
# def non_max_suppression(box_mask_list,
#                         max_output_size=10000,
#                         iou_threshold=1.0,
#                         score_threshold=-10.0):
#     """Non maximum suppression.
#
#   This op greedily selects a subset of detection bounding boxes, pruning
#   away boxes that have high IOU (intersection over union) overlap (> thresh)
#   with already selected boxes. In each iteration, the detected bounding box with
#   highest score in the available pool is selected.
#
#   Args:
#     box_mask_list: np_box_mask_list.BoxMaskList holding N boxes.  Must contain
#       a 'scores' field representing detection scores. All scores belong to the
#       same class.
#     max_output_size: maximum number of retained boxes
#     iou_threshold: intersection over union threshold.
#     score_threshold: minimum score threshold. Remove the boxes with scores
#                      less than this value. Default value is set to -10. A very
#                      low threshold to pass pretty much all the boxes, unless
#                      the user sets a different score threshold.
#
#   Returns:
#     an np_box_mask_list.BoxMaskList holding M boxes where M <= max_output_size
#
#   Raises:
#     ValueError: if 'scores' field does not exist
#     ValueError: if threshold is not in [0, 1]
#     ValueError: if max_output_size < 0
#   """
#     if not box_mask_list.has_field('scores'):
#         raise ValueError('Field scores does not exist')
#     if iou_threshold < 0. or iou_threshold > 1.0:
#         raise ValueError('IOU threshold must be in [0, 1]')
#     if max_output_size < 0:
#         raise ValueError('max_output_size must be bigger than 0.')
#
#     box_mask_list = filter_scores_greater_than(box_mask_list, score_threshold)
#     if box_mask_list.num_boxes() == 0:
#         return box_mask_list
#
#     box_mask_list = sort_by_field(box_mask_list, 'scores')
#
#     # Prevent further computation if NMS is disabled.
#     if iou_threshold == 1.0:
#         if box_mask_list.num_boxes() > max_output_size:
#             selected_indices = np.arange(max_output_size)
#             return gather(box_mask_list, selected_indices)
#         else:
#             return box_mask_list
#
#     masks = box_mask_list.get_masks()
#     num_masks = box_mask_list.num_boxes()
#
#     # is_index_valid is True only for all remaining valid boxes,
#     is_index_valid = np.full(num_masks, 1, dtype=bool)
#     selected_indices = []
#     num_output = 0
#     for i in range(num_masks):
#         if num_output < max_output_size:
#             if is_index_valid[i]:
#                 num_output += 1
#                 selected_indices.append(i)
#                 is_index_valid[i] = False
#                 valid_indices = np.where(is_index_valid)[0]
#                 if valid_indices.size == 0:
#                     break
#
#                 intersect_over_union = np_mask_ops_iou(
#                     np.expand_dims(masks[i], axis=0), masks[valid_indices])
#                 intersect_over_union = np.squeeze(intersect_over_union, axis=0)
#                 is_index_valid[valid_indices] = np.logical_and(
#                     is_index_valid[valid_indices],
#                     intersect_over_union <= iou_threshold)
#     return gather(box_mask_list, np.array(selected_indices))
#
#
# def np_box_list_ops_gather(boxlist, indices, fields=None):
#     """Gather boxes from BoxList according to indices and return new BoxList.
#
#   By default, gather returns boxes corresponding to the input index list, as
#   well as all additional fields stored in the boxlist (indexing into the
#   first dimension).  However one can optionally only gather from a
#   subset of fields.
#
#   Args:
#     boxlist: BoxList holding N boxes
#     indices: a 1-d numpy array of type int_
#     fields: (optional) list of fields to also gather from.  If None (default),
#         all fields are gathered from.  Pass an empty fields list to only gather
#         the box coordinates.
#
#   Returns:
#     subboxlist: a BoxList corresponding to the subset of the input BoxList
#         specified by indices
#
#   Raises:
#     ValueError: if specified field is not contained in boxlist or if the
#         indices are not of type int_
#   """
#     if indices.size:
#         if np.amax(indices) >= boxlist.num_boxes() or np.amin(indices) < 0:
#             raise ValueError('indices are out of valid range.')
#     subboxlist = BoxList(boxlist.get()[indices, :])
#     if fields is None:
#         fields = boxlist.get_extra_fields()
#     for field in fields:
#         extra_field_data = boxlist.get_field(field)
#         subboxlist.add_field(field, extra_field_data[indices, ...])
#     return subboxlist
#
#
# def gather(box_mask_list, indices, fields=None):
#     """Gather boxes from np_box_mask_list.BoxMaskList according to indices.
#
#   By default, gather returns boxes corresponding to the input index list, as
#   well as all additional fields stored in the box_mask_list (indexing into the
#   first dimension).  However one can optionally only gather from a
#   subset of fields.
#
#   Args:
#     box_mask_list: np_box_mask_list.BoxMaskList holding N boxes
#     indices: a 1-d numpy array of type int_
#     fields: (optional) list of fields to also gather from.  If None (default),
#         all fields are gathered from.  Pass an empty fields list to only gather
#         the box coordinates.
#
#   Returns:
#     subbox_mask_list: a np_box_mask_list.BoxMaskList corresponding to the subset
#         of the input box_mask_list specified by indices
#
#   Raises:
#     ValueError: if specified field is not contained in box_mask_list or if the
#         indices are not of type int_
#   """
#     if fields is not None:
#         if 'masks' not in fields:
#             fields.append('masks')
#     return box_list_to_box_mask_list(
#         np_box_list_ops_gather(
#             boxlist=box_mask_list, indices=indices, fields=fields))
#
#
# def box_list_to_box_mask_list(boxlist):
#     """Converts a BoxList containing 'masks' into a BoxMaskList.
#
#   Args:
#     boxlist: An np_box_list.BoxList object.
#
#   Returns:
#     An np_box_mask_list.BoxMaskList object.
#
#   Raises:
#     ValueError: If boxlist does not contain `masks` as a field.
#   """
#     if not boxlist.has_field('masks'):
#         raise ValueError('boxlist does not contain mask field.')
#     box_mask_list = BoxMaskList(
#         box_data=boxlist.get(),
#         mask_data=boxlist.get_field('masks'))
#     extra_fields = boxlist.get_extra_fields()
#     for key in extra_fields:
#         if key != 'masks':
#             box_mask_list.data[key] = boxlist.get_field(key)
#     return box_mask_list
#
#
# def sort_by_field(box_mask_list, field,
#                   order=SortOrder.DESCEND):
#     """Sort boxes and associated fields according to a scalar field.
#
#   A common use case is reordering the boxes according to descending scores.
#
#   Args:
#     box_mask_list: BoxMaskList holding N boxes.
#     field: A BoxMaskList field for sorting and reordering the BoxMaskList.
#     order: (Optional) 'descend' or 'ascend'. Default is descend.
#
#   Returns:
#     sorted_box_mask_list: A sorted BoxMaskList with the field in the specified
#       order.
#   """
#     return box_list_to_box_mask_list(
#         sort_by_field(
#             boxlist=box_mask_list, field=field, order=order))
#
#
# def filter_scores_greater_than(box_mask_list, thresh):
#     """Filter to keep only boxes and masks with score exceeding a given threshold.
#
#   This op keeps the collection of boxes and masks whose corresponding scores are
#   greater than the input threshold.
#
#   Args:
#     box_mask_list: BoxMaskList holding N boxes and masks.  Must contain a
#       'scores' field representing detection scores.
#     thresh: scalar threshold
#
#   Returns:
#     a BoxMaskList holding M boxes and masks where M <= N
#
#   Raises:
#     ValueError: if box_mask_list not a np_box_mask_list.BoxMaskList object or
#       if it does not have a scores field
#   """
#     if not isinstance(box_mask_list, BoxMaskList):
#         raise ValueError('box_mask_list must be a BoxMaskList')
#     if not box_mask_list.has_field('scores'):
#         raise ValueError('input box_mask_list must have \'scores\' field')
#     scores = box_mask_list.get_field('scores')
#     if len(scores.shape) > 2:
#         raise ValueError('Scores should have rank 1 or 2')
#     if len(scores.shape) == 2 and scores.shape[1] != 1:
#         raise ValueError('Scores should have rank 1 or have shape '
#                          'consistent with [None, 1]')
#     high_score_indices = np.reshape(np.where(np.greater(scores, thresh)),
#                                     [-1]).astype(np.int32)
#     return gather(box_mask_list, high_score_indices)


# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True
#
#
# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()


#
# def all_gather(data):
#     """
#     Run all_gather on arbitrary picklable data (not necessarily tensors)
#     Args:
#         data: any picklable object
#     Returns:
#         list[data]: list of data gathered from each rank
#     """
#     world_size = get_world_size()
#     if world_size == 1:
#         return [data]
#
#     # serialized to a Tensor
#     buffer = pickle.dumps(data)
#     storage = torch.ByteStorage.from_buffer(buffer)
#     tensor = torch.ByteTensor(storage).to("cuda")
#
#     # obtain Tensor size of each rank
#     local_size = torch.tensor([tensor.numel()], device="cuda")
#     size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
#     dist.all_gather(size_list, local_size)
#     size_list = [int(size.item()) for size in size_list]
#     max_size = max(size_list)
#
#     # receiving Tensor from all ranks
#     # we pad the tensor because torch all_gather does not support
#     # gathering tensors of different shapes
#     tensor_list = []
#     for _ in size_list:
#         tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
#     if local_size != max_size:
#         padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
#         tensor = torch.cat((tensor, padding), dim=0)
#     dist.all_gather(tensor_list, tensor)
#
#     data_list = []
#     for size, tensor in zip(size_list, tensor_list):
#         buffer = tensor.cpu().numpy().tobytes()[:size]
#         data_list.append(pickle.loads(buffer))
#
#     return data_list

#
# class CocoEvaluator(object):
#     def __init__(self, coco_gt, iou_types):
#         assert isinstance(iou_types, (list, tuple))
#         coco_gt = copy.deepcopy(coco_gt)
#         self.coco_gt = coco_gt
#
#         self.iou_types = iou_types
#         self.coco_eval = {}
#         for iou_type in iou_types:
#             self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
#
#         self.img_ids = []
#         self.eval_imgs = {k: [] for k in iou_types}
#
#     def update(self, results):
#         img_ids = list(np.unique([res['image_id'] for res in results]))
#         self.img_ids.extend(img_ids)
#
#         for iou_type in self.iou_types:
#             # suppress pycocotools prints
#             with open(os.devnull, 'w') as devnull:
#                 with contextlib.redirect_stdout(devnull):
#                     coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
#             coco_eval = self.coco_eval[iou_type]
#
#             coco_eval.cocoDt = coco_dt
#             coco_eval.params.imgIds = list(img_ids)
#             img_ids, eval_imgs = evaluate(coco_eval)
#
#             self.eval_imgs[iou_type].append(eval_imgs)
#
#     def synchronize_between_processes(self):
#         for iou_type in self.iou_types:
#             self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
#             create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])
#
#     def accumulate(self):
#         for coco_eval in self.coco_eval.values():
#             coco_eval.accumulate()
#
#     def summarize(self):
#         for iou_type, coco_eval in self.coco_eval.items():
#             print("IoU metric: {}".format(iou_type))
#             coco_eval.summarize()

#
# def merge(img_ids, eval_imgs):
#     all_img_ids = all_gather(img_ids)
#     all_eval_imgs = all_gather(eval_imgs)
#
#     merged_img_ids = []
#     for p in all_img_ids:
#         merged_img_ids.extend(p)
#
#     merged_eval_imgs = []
#     for p in all_eval_imgs:
#         merged_eval_imgs.append(p)
#
#     merged_img_ids = np.array(merged_img_ids)
#     merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)
#
#     # keep only unique (and in sorted order) images
#     merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
#     merged_eval_imgs = merged_eval_imgs[..., idx]
#
#     return merged_img_ids, merged_eval_imgs

#
# def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
#     img_ids, eval_imgs = merge(img_ids, eval_imgs)
#     img_ids = list(img_ids)
#     eval_imgs = list(eval_imgs.flatten())
#
#     coco_eval.evalImgs = eval_imgs
#     coco_eval.params.imgIds = img_ids
#     coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
#
#
# def evaluate(self):
#     # tic = time.time()
#     # print('Running per image evaluation...')
#     p = self.params
#     # add backward compatibility if useSegm is specified in params
#     if p.useSegm is not None:
#         p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
#         print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
#     # print('Evaluate annotation type *{}*'.format(p.iouType))
#     p.imgIds = list(np.unique(p.imgIds))
#     if p.useCats:
#         p.catIds = list(np.unique(p.catIds))
#     p.maxDets = sorted(p.maxDets)
#     self.params = p
#
#     self._prepare()
#     # loop through images, area range, max detection number
#     catIds = p.catIds if p.useCats else [-1]
#
#     if p.iouType == 'segm' or p.iouType == 'bbox':
#         computeIoU = self.computeIoU
#     elif p.iouType == 'keypoints':
#         computeIoU = self.computeOks
#     self.ious = {
#         (imgId, catId): computeIoU(imgId, catId)
#         for imgId in p.imgIds
#         for catId in catIds}
#
#     evaluateImg = self.evaluateImg
#     maxDet = p.maxDets[-1]
#     evalImgs = [
#         evaluateImg(imgId, catId, areaRng, maxDet)
#         for catId in catIds
#         for areaRng in p.areaRng
#         for imgId in p.imgIds
#     ]
#     # this is NOT in the pycocotools code, but could be done outside
#     evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
#     self._paramsEval = copy.deepcopy(self.params)
#     # toc = time.time()
#     # print('DONE (t={:0.2f}s).'.format(toc-tic))
#     return p.imgIds, evalImgs
#


_resize_methods = {'nearest': cv2.INTER_NEAREST,
                   'bilinear': cv2.INTER_LINEAR,
                   'area': cv2.INTER_AREA,
                   'cubic': cv2.INTER_CUBIC,
                   'lanczos': cv2.INTER_LANCZOS4,
                   }


class AspectPreservingResizeWithPad(object):
    def __init__(self, new_height, new_width, pad=True, pad_value=0, resize_method='area',
                 resize_label=False, label_resize_method='nearest', label_pad_value=255):
        self.name = "AspectPreservingResizeWithPad"
        self.new_height = new_height
        self.new_width = new_width
        self.pad = pad
        self.pad_value = pad_value
        self.resize_method = _resize_methods[resize_method]
        self.resize_label = resize_label
        self.label_resize_method = _resize_methods[label_resize_method]
        self.label_pad_value = label_pad_value

    def __call__(self, img, label):
        try:
            resize_ratio = max(img.shape[0] / self.new_height, img.shape[1] / self.new_width)
        except:
            print('ar')
        height_tag = int(np.round(img.shape[0] / resize_ratio))
        width_tag = int(np.round(img.shape[1] / resize_ratio))
        pad_values = ((int((self.new_height - height_tag) / 2), int((self.new_height - height_tag) / 2 + 0.5)),
                      (int((self.new_width - width_tag) / 2), int((self.new_width - width_tag) / 2 + 0.5)),
                      (0, 0))

        resized_img = cv2.resize(img, (width_tag, height_tag), interpolation=self.resize_method)
        if not self.pad:
            padded_img = resized_img
        elif isinstance(self.pad_value, (float, int)):
            padded_img = np.pad(resized_img, pad_values, constant_values=self.pad_value)
        else:
            padded_img = np.stack([np.pad(resized_img[..., 0], pad_values[:2], constant_values=self.pad_value[0]),
                                   np.pad(resized_img[..., 1], pad_values[:2], constant_values=self.pad_value[1]),
                                   np.pad(resized_img[..., 2], pad_values[:2], constant_values=self.pad_value[2])], 2)

        if self.resize_label and label is not None:
            if type(label) == dict and 'masks' in label.keys():
                masks = label['masks']
                masks = np.transpose(masks, (1, 2, 0))  # CHW -> HWC
                resized_masks = cv2.resize(masks, (width_tag, height_tag), interpolation=self.label_resize_method)
                resized_masks = np.expand_dims(resized_masks, -1) if len(resized_masks.shape) == 2 else resized_masks
                padded_masks = np.pad(resized_masks, pad_values[:len(label['masks'].shape)],
                                      constant_values=self.label_pad_value) if self.pad else resized_masks
                padded_masks = np.transpose(padded_masks, (2, 0, 1))  # HWC -> CHW
                label['masks'] = padded_masks > 0.5
                padded_label = label
            else:
                resized_label = cv2.resize(label, (width_tag, height_tag), interpolation=self.label_resize_method)
                padded_label = np.pad(resized_label, pad_values[:len(label.shape)],
                                      constant_values=self.label_pad_value) if self.pad else resized_label

        else:
            padded_label = label

        return padded_img, padded_label


class Normalize(object):

    def __init__(self, mean, std):
        self.name = 'Normalization'
        self.mean = mean
        self.std = std

    def __call__(self, img, label):
        return (img - self.mean) / self.std, label


class Transpose(object):
    AXES = (2, 0, 1)

    def __init__(self, axes=AXES):
        """
        :param axes:  list of ints, optional. By default, reverse the dimensions, otherwise permute
        the axes according to the values given.
        """
        self.name = 'Transpose'
        self.axes = axes

    def __call__(self, img, label):
        return np.transpose(img, self.axes), label


def get_transform():
    custom_transforms = [
        # transforms.ToTensor(),
        # transforms.Resize((640, 640)),
        # transforms.RandomCrop((640, 640)),
        AspectPreservingResizeWithPad(640, 640, pad_value=114),
        Normalize(0.0, 255.0),
        Transpose()
        # transforms.ToTensor()
    ]
    return torchvision.transforms.Compose(custom_transforms)


pre_process = [AspectPreservingResizeWithPad(640, 640, pad_value=114),
               Normalize(0.0, 255.0),
               Transpose()]


#
# def coco80_to_coco91(x: np.ndarray) -> np.ndarray:
#     """
#     Converts COCO 80-class indices to COCO 91-class indices.
#
#     Args:
#         x (numpy.ndarray): An array of COCO 80-class indices.
#
#     Returns:
#         numpy.ndarray: An array of corresponding COCO 91-class indices.
#     """
#     coco91Indexs = np.array(
#         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
#          35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
#          63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90])
#     return coco91Indexs[x.astype(np.int32)]


# def clip_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
#     """
#     Clip bounding boxes to stay within the image boundaries.
#
#     Args:
#         boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
#         h (int): Height of the image.
#         w (int): Width of the image.
#
#     Returns:
#         numpy.ndarray: Clipped bounding boxes.
#     """
#     boxes[..., 0] = np.clip(boxes[..., 0], a_min=0, a_max=h)
#     boxes[..., 1] = np.clip(boxes[..., 1], a_min=0, a_max=w)
#     boxes[..., 2] = np.clip(boxes[..., 2], a_min=0, a_max=h)
#     boxes[..., 3] = np.clip(boxes[..., 3], a_min=0, a_max=w)
#     return boxes
#
#
# def scale_boxes(boxes: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int,
#                 preserve_aspect_ratio: bool) -> np.ndarray:
#     """
#     Scale and offset bounding boxes based on model output size and original image size.
#
#     Args:
#         boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
#         h_image (int): Original image height.
#         w_image (int): Original image width.
#         h_model (int): Model output height.
#         w_model (int): Model output width.
#         preserve_aspect_ratio (bool): Whether to preserve image aspect ratio during scaling
#
#     Returns:
#         numpy.ndarray: Scaled and offset bounding boxes.
#     """
#     deltaH, deltaW = 0, 0
#     H, W = h_model, w_model
#     scale_H, scale_W = h_image / H, w_image / W
#
#     if preserve_aspect_ratio:
#         scale_H = scale_W = max(h_image / H, w_image / W)
#         H_tag = int(np.round(h_image / scale_H))
#         W_tag = int(np.round(w_image / scale_W))
#         deltaH, deltaW = int((H - H_tag) / 2), int((W - W_tag) / 2)
#
#     # Scale and offset boxes
#     boxes[..., 0] = (boxes[..., 0] * H - deltaH) * scale_H
#     boxes[..., 1] = (boxes[..., 1] * W - deltaW) * scale_W
#     boxes[..., 2] = (boxes[..., 2] * H - deltaH) * scale_H
#     boxes[..., 3] = (boxes[..., 3] * W - deltaW) * scale_W
#
#     # Clip boxes
#     boxes = clip_boxes(boxes, h_image, w_image)
#
#     return boxes
#
#
# def format_results(outputs: List, img_ids: List, orig_img_dims: List, output_resize: Dict) -> List[Dict]:
#     """
#     Format model outputs into a list of detection dictionaries.
#
#     Args:
#         outputs (list): List of model outputs, typically containing bounding boxes, scores, and labels.
#         img_ids (list): List of image IDs corresponding to each output.
#         orig_img_dims (list): List of tuples representing the original image dimensions (h, w) for each output.
#         output_resize (Dict): Contains the resize information to map between the model's
#                  output and the original image dimensions.
#
#     Returns:
#         list: A list of detection dictionaries, each containing information about the detected object.
#     """
#     detections = []
#     h_model, w_model = output_resize['shape']
#     preserve_aspect_ratio = output_resize['aspect_ratio_preservation']
#
#     # Process model outputs and convert to detection format
#     for idx, output in enumerate(outputs):
#         image_id = img_ids[idx]
#         scores = output[1].detach().cpu().numpy().squeeze()  # Extract scores
#         labels = (coco80_to_coco91(
#             output[2].detach().cpu().numpy())).squeeze()  # Convert COCO 80-class indices to COCO 91-class indices
#         boxes = output[0].detach().cpu().numpy().squeeze()  # Extract bounding boxes
#         boxes = scale_boxes(boxes, orig_img_dims[idx][0], orig_img_dims[idx][1], h_model, w_model,
#                             preserve_aspect_ratio)
#
#         for score, label, box in zip(scores, labels, boxes):
#             detection = {
#                 "image_id": image_id,
#                 "category_id": label,
#                 "bbox": [box[1], box[0], box[3] - box[1], box[2] - box[0]],
#                 "score": score
#             }
#             detections.append(detection)
#
#     return detections
#
# # COCO evaluation class
# class CocoEval:
#     def __init__(self, path2json: str, output_resize: Dict = None, gt_boxes_format=BoxFormat.XMIN_YMIN_W_H,
#                  num_categories=80, iou_list=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
#         """
#         Initialize the CocoEval class.
#
#         Args:
#             path2json (str): Path to the COCO JSON file containing ground truth annotations.
#             output_resize (Dict): Contains the resize information to map between the model's output and the original
#              image dimensions. The dict consists of:
#                   {"shape": (height, weight),
#                    "aspect_ratio_preservation": bool}
#         """
#         # Load ground truth annotations
#         self.coco_gt = COCO(path2json)
#
#         # A list of reformatted model outputs
#         self.all_detections = []
#
#         # Resizing information to map between the model's output and the original image dimensions
#         self.output_resize = output_resize if output_resize else {'shape': (1, 1), 'aspect_ratio_preservation': False}
#         self.gt_boxes_format = gt_boxes_format
#         self.num_categories = num_categories
#         self.iou_list = iou_list
#         self.evaluators = [ObjectDetectionEvaluation(self.num_categories, matching_iou_threshold=iou) for iou in
#                            self.iou_list]
#         self.batch_counter = 0
#
#     # def add_batch_detections(self, outputs: Tuple[List, List, List, List], targets: List[Dict]):
#     #     """
#     #     Add batch detections to the evaluation.
#     #
#     #     Args:
#     #         outputs (list): List of model outputs, typically containing bounding boxes, scores, and labels.
#     #         targets (list): List of ground truth annotations for the batch.
#     #     """
#     #     img_ids, _outs = [], []
#     #     orig_img_dims = []
#     #     for idx, t in enumerate(targets):
#     #         if len(t) > 0:
#     #             img_ids.append(t[0]['image_id'])
#     #             orig_img_dims.append(t[0]['orig_img_dims'])
#     #             _outs.append([outputs[0][idx], outputs[1][idx], outputs[2][idx], outputs[3][idx]])
#     #
#     #     batch_detections = format_results(_outs, img_ids, orig_img_dims, self.output_resize)
#     #
#     #     self.all_detections.extend(batch_detections)
#
#     def add_batch_detections(self, output_annotations, target_annotations):
#         for j in range(len(target_annotations)):
#             gt_ann = target_annotations[j]
#             out_ann = output_annotations[j]
#             if gt_ann is None:
#                 continue
#             # nd array at runner
#             if len(gt_ann['boxes']) > 0:
#                 orig_width = target_annotations[j]['orig_img_dims'][1]
#                 orig_height = target_annotations[j]['orig_img_dims'][0]
#                 gt_boxes = convert_to_ymin_xmin_ymax_xmax_format(gt_ann['boxes'],
#                                                                  self.gt_boxes_format)
#                 gt_boxes = apply_normalization(gt_boxes, orig_width, orig_height,
#                                                BoxFormat.YMIM_XMIN_YMAX_XMAX)
#                 [evaluator.add_single_ground_truth_image_info(gt_ann['image_id'],
#                                                               gt_boxes,
#                                                               np.array(gt_ann['classes']))
#                  for evaluator in self.evaluators]
#
#                 if len(out_ann['boxes']) > 0:
#                     if False:
#                         out_ann['boxes'] = normalize_boxes(out_ann['boxes'], orig_height, orig_width, scale_factor,
#                                                            H, W, preserve_aspect_ratio)
#                     [evaluator.add_single_detected_image_info(gt_ann['image_id'], out_ann['boxes'],
#                                                               out_ann['scores'],
#                                                               np.array(out_ann['classes']))
#                      for evaluator in self.evaluators]
#
#         self.batch_counter += 1
#
#     def result(self) -> List[float]:
#         """
#         Calculate and print evaluation results.
#
#         Returns:
#             list: COCO evaluation statistics.
#         """
#         # Initialize COCO evaluation object
#         self.coco_dt = self.coco_gt.loadRes(self.all_detections)
#         coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox')
#
#         # Run evaluation
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()
#
#         # Print mAP results
#         print("mAP: {:.4f}".format(coco_eval.stats[0]))
#
#         return coco_eval.stats
#
#     def reset(self):
#         """
#         Reset the list of detections to prepare for a new evaluation.
#         """
#         self.all_detections = []
#
#     def summarize_results(self, is_multiprocess=False):
#         results = [evaluator.evaluate()[1] for evaluator in self.evaluators]
#         return np.mean(np.array(results))
#
#
# def normalize_boxes(boxes, orig_height, orig_width, scale_factor, H, W, preserve_aspect_ratio):
#     boxes = boxes * scale_factor
#     boxes = scale_boxes(boxes, orig_height, orig_width, H, W, preserve_aspect_ratio)
#     boxes = apply_normalization(boxes, orig_width, orig_height, BoxFormat.YMIM_XMIN_YMAX_XMAX)
#     return boxes


def load_and_preprocess_image(image_path: str, preprocess: Callable) -> np.ndarray:
    """
    Load and preprocess an image from a given file path.

    Args:
        image_path (str): Path to the image file.
        preprocess (function): Preprocessing function to apply to the loaded image.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    image = cv2.imread(image_path)
    image = preprocess(image)
    return image


def coco_dataset_generator(dataset_folder: str, annotation_file: str, preprocess: Callable,
                           batch_size: int = 1) -> Tuple:
    # Load COCO annotations from a JSON file (e.g., 'annotations.json')
    with open(annotation_file, 'r') as f:
        coco_annotations = json.load(f)

    # Initialize a dictionary to store annotations grouped by image ID
    annotations_by_image = {}

    # Iterate through the annotations and group them by image ID
    for annotation in coco_annotations['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Initialize a list to collect images and annotations for the current batch
    batch_images = []
    batch_annotations = []
    total_images = len(coco_annotations['images'])

    # Iterate through the images and create a list of tuples (image, annotations)
    for image_count, image_info in enumerate(coco_annotations['images']):
        image_id = image_info['id']
        # Load and preprocess the image (you can use your own image loading logic)
        image = load_and_preprocess_image(os.path.join(dataset_folder, image_info['file_name']), preprocess)
        annotations = annotations_by_image.get(image_id, [])
        if len(annotations) > 0:
            boxes = np.asarray([ia['bbox'] for ia in annotations])
            classes = np.asarray([ia['category_id'] for ia in annotations])
            label = {'image_id': int(image_id), 'boxes': boxes, 'classes': classes,
                     'orig_img_dims': (image_info['height'], image_info['width'])}
            # annotations[0]['orig_img_dims'] = (image_info['height'], image_info['width'])

            # Add the image and annotations to the current batch
            batch_images.append(image)
            batch_annotations.append(label)
            # batch_annotations.append(annotations)

            # Check if the current batch is of the desired batch size
            if len(batch_images) == batch_size:
                # Yield the current batch
                # yield np.array(batch_images), batch_annotations
                # batch_images = preprocess(np.array(batch_images))
                yield torch.stack(batch_images), batch_annotations

                # Reset the batch lists for the next batch
                batch_images = []
                batch_annotations = []

        # After processing all images, yield any remaining images in the last batch
        if len(batch_images) > 0 and (total_images == image_count + 1):
            # yield np.array(batch_images), batch_annotations
            yield batch_images, batch_annotations


def coco_evaluate(model: Any, preprocess: Callable, dataset_folder: str, annotation_file: str, batch_size: int,
                  output_resize: tuple, device='cuda', original=False) -> dict:
    """
    Evaluate a model on the COCO dataset.

    Args:
    - model (Any): The model to evaluate.
    - preprocess (Callable): Preprocessing function to be applied to images.
    - dataset_folder (str): Path to the folder containing COCO dataset images.
    - annotation_file (str): Path to the COCO annotation file.
    - batch_size (int): Batch size for evaluation.
    - output_resize (tuple): Tuple representing the output size after resizing.

    Returns:
    - dict: Evaluation results.

    """
    # Load COCO evaluation set
    # val_dataset = coco_dataset_generator(dataset_folder=dataset_folder,
    #                                      annotation_file=annotation_file,
    #                                      preprocess=preprocess,
    #                                      batch_size=batch_size)
    # val_dataset = dataset_generator(dataset_folder=dataset_folder,
    #                                 annotation_file=annotation_file,
    #                                 preprocess=preprocess,
    #                                 batch_size=batch_size)
    input_dataset = CoCoDataset(dataset_path=dataset_folder, _gt_annotations_path=annotation_file,
                                preprocessing=preprocess)

    # Initialize the evaluation metric object
    # coco_metric = CocoEval(annotation_file, output_resize)
    coco_metric = PostProcessGoogleMeanAveragePrecision(model_input_shape=output_resize['shape'])
    num_predicted_images = 0
    NUM_IMAGES_TO_DISPLAY = 1000

    # Iterate and the evaluation set
    # for batch_idx, (images, targets) in enumerate(val_dataset):
    for batch_idx, (_, images, targets) in enumerate(tqdm(input_dataset.data_generator(batch_size))):

        effective_batch_size = images.shape[0]

        # Run inference on the batch
        if original:
            outputs = model(images.type(torch.cuda.HalfTensor))
        else:
            outputs = model(images.to(device, dtype=torch.float))

        coco_metric.add_batch_results(outputs, targets)
        num_predicted_images += effective_batch_size
        if num_predicted_images % NUM_IMAGES_TO_DISPLAY == 0:
            print(coco_metric.summarize_results())
    res = coco_metric.summarize_results()
    print(res)
    # postprocess
    # box
    # nms
    # image_shapes = [x['orig_img_dims'] for x in targets]  # original image shape
    # np_out = (np.asarray(outputs[0].detach().cpu()), np.asarray(outputs[1].detach().cpu()))
    # preds = post_process(np_out, image_shapes=image_shapes)

    # Add the model outputs to metric object (a dictionary of outputs after postprocess: boxes, scores & classes)
    # coco_metric.add_batch_detections(preds, targets)
    # if (batch_idx + 1) % 100 == 0:
    #     print(f'processed {(batch_idx + 1) * batch_size} images')

    # return coco_metric.result()
    # return coco_metric.summarize_results()

# def box_decoding_YOLO_V8(prediction,
#                          conf_thres=0.001,
#                          nc=80,  # number of classes
#                          ):
#     if isinstance(prediction,
#                   (list, tuple)):  # YOLOv8 model in validation model, output = (bbox, classes)
#         feat_sizes = np.array([80, 40, 20])
#         stride_sizes = np.array([8, 16, 32])
#         anchors, strides = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))
#         dbox = dist2bbox_yolo_v8(prediction[0], np.expand_dims(anchors, 0), xywh=True, dim=1) * strides
#         if len(prediction) == 4:  # include masks for instance segmentation
#             prediction = np.concatenate((dbox, prediction[1], prediction[2]), 1)
#         else:
#             prediction = np.concatenate((dbox, prediction[1]), 1)
#
#     mi = 4 + nc  # mask start index
#
#     xc = np.amax(prediction[:, 4:mi], 1) > conf_thres
#
#     z = []
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         zc = np.transpose(x)[xc[xi]]
#         z.append(np.c_[zc[:, 0:4], np.ones(zc.shape[0]), zc[:, 4:]])
#
#     return z


# def make_anchors_yolo_v8(feats, strides, grid_cell_offset=0.5):
#     """Generate anchors from features."""
#     anchor_points, stride_tensor = [], []
#     assert feats is not None
#     for i, stride in enumerate(strides):
#         h, w = feats[i], feats[i]
#         sx = np.arange(stop=w) + grid_cell_offset  # shift x
#         sy = np.arange(stop=h) + grid_cell_offset  # shift y
#         sy, sx = np.meshgrid(sy, sx, indexing='ij')
#         anchor_points.append(np.stack((sx, sy), -1).reshape((-1, 2)))
#         stride_tensor.append(np.full((h * w, 1), stride))
#     return np.concatenate(anchor_points), np.concatenate(stride_tensor)
#
#
# def dist2bbox_yolo_v8(distance, anchor_points, xywh=True, dim=-1):
#     """Transform distance(ltrb) to box(xywh or xyxy)."""
#     lt, rb = np.split(distance, 2, axis=dim)
#     x1y1 = anchor_points - lt
#     x2y2 = anchor_points + rb
#     if xywh:
#         c_xy = (x1y1 + x2y2) / 2
#         wh = x2y2 - x1y1
#         return np.concatenate((c_xy, wh), dim)  # xywh bbox
#     return np.concatenate((x1y1, x2y2), dim)  # xyxy bbox
#
#
# def post_process(output_annotations, image_shapes, conf_thres=0.001, iou_thres=0.65, normalize_boxes=True, min_wh=2,
#                  max_wh=7680, max_nms_dets=5000, H=640, W=640, nc=80, background=False, preserve_aspect_ratio=True):
#     outputs_decoded = box_decoding_YOLO_V8(output_annotations, nc=nc)
#
#     ############################################################
#     # Post processing for each input image
#     ############################################################
#     # Note: outputs_decoded shape is [Batch,num_anchors*Detections,(4+1+num_categories)]
#     post_processed_outputs = []
#     for i, x in enumerate(outputs_decoded):
#         # ----------------------------------------
#         # Filter by score and width-height
#         # ----------------------------------------
#         scores = x[..., 4]
#         wh = x[..., 2:4]
#         valid_indexs = (scores > conf_thres) & ((wh > min_wh).any(1)) & ((wh < max_wh).any(1))
#         x = x[valid_indexs]
#
#         # ----------------------------------------
#         # Taking Best class only
#         # ----------------------------------------
#         x[..., 5:] *= x[..., 4:5]  # compute confidence per class (class_score * object_score)
#         conf = np.max(x[:, 5:], axis=1, keepdims=True)
#         classes_id = np.argmax(x[:, 5:], axis=1, keepdims=True)
#
#         # Change boxes format from [x_c,y_c,w,h] to [y_min,x_min,y_max,x_max]
#         boxes = convert_to_ymin_xmin_ymax_xmax_format(x[..., :4], BoxFormat.XC_YC_W_H)
#         x = np.concatenate((boxes, conf, classes_id), axis=1)[conf.reshape(-1) > conf_thres]
#
#         # --------------------------- #
#         # NMS
#         # --------------------------- #
#         x = x[np.argsort(-x[:, 4])[:max_nms_dets]]  # sort by confidence from high to low
#         offset = x[..., 5:6] * np.maximum(H, W)
#         boxes_offset, scores = x[..., :4] + offset, x[..., 4]  # boxes with offset by class
#         valid_indexs = nms(boxes_offset, scores, iou_thres)
#         x = x[valid_indexs]
#
#         # --------------------------- #
#         # Boxes process: scale shapes according to original input image and normalize boxes
#         # --------------------------- #
#         h_image, w_image = image_shapes[i][:2]
#         boxes = scale_boxes(x[..., :4], h_image, w_image, H, W, preserve_aspect_ratio)
#         boxes = apply_normalization(boxes, w_image, h_image,
#                                     BoxFormat.YMIM_XMIN_YMAX_XMAX) if normalize_boxes else boxes
#
#         # --------------------------- #
#         # Classes process
#         # --------------------------- #
#         # convert classes from coco80 to coco91 to match labels
#         classes = coco80_to_coco91(x[..., 5]) if nc == 80 else x[..., 5]
#         classes -= 0 if background else 1
#
#         # --------------------------- #
#         # Scores
#         # --------------------------- #
#         scores = x[..., 4]
#
#         # Add result
#         post_processed_outputs.append({'boxes': boxes, 'classes': classes, 'scores': scores})
#
#     return post_processed_outputs
#
#
# def convert_to_ymin_xmin_ymax_xmax_format(boxes, orig_format: BoxFormat):
#     """
#     changes the box from one format to another (XMIN_YMIN_W_H --> YMIM_XMIN_YMAX_XMAX )
#     also support in same format mode (returns the same format)
#
#     :param boxes:
#     :param orig_format:
#     :return: box in format YMIM_XMIN_YMAX_XMAX
#     """
#     if len(boxes) == 0:
#         return boxes
#     elif orig_format == BoxFormat.YMIM_XMIN_YMAX_XMAX:
#         return boxes
#     elif orig_format == BoxFormat.XMIN_YMIN_W_H:
#         boxes[:, 2] += boxes[:, 0]  # convert width to xmax
#         boxes[:, 3] += boxes[:, 1]  # convert height to ymax
#         boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
#         boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
#         return boxes
#     elif orig_format == BoxFormat.XMIM_YMIN_XMAX_YMAX:
#         boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
#         boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
#         return boxes
#     elif orig_format == BoxFormat.XC_YC_W_H:
#         new_boxes = np.copy(boxes)
#         new_boxes[:, 0] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
#         new_boxes[:, 1] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
#         new_boxes[:, 2] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
#         new_boxes[:, 3] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
#         return new_boxes
#     else:
#         raise Exception("Unsupported boxes format")
#
#
# def convert_to_xmin_ymin_width_height_format(boxes, orig_format: BoxFormat):
#     """
#     changes the box from one format to another (YMIM_XMIN_YMAX_XMAX --> XMIN_YMIN_W_H)
#     also support in same format mode (returns the same format)
#
#     :param boxes:
#     :param orig_format:
#     :return: box in format XMIN_YMIN_W_H
#     """
#     if len(boxes) == 0:
#         return boxes
#     elif orig_format == BoxFormat.XMIN_YMIN_W_H:
#         return boxes
#     elif orig_format == BoxFormat.YMIM_XMIN_YMAX_XMAX:
#         boxes[:, 2] -= boxes[:, 0]  # convert ymax to height
#         boxes[:, 3] -= boxes[:, 1]  # convert xmax to width
#         boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap ymin, xmin columns
#         boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap height, width columns
#         return boxes
#     elif orig_format == BoxFormat.XMIM_YMIN_XMAX_YMAX:
#         boxes[:, 2] -= boxes[:, 0]  # convert ymax to height
#         boxes[:, 3] -= boxes[:, 1]  # convert xmax to width
#         return boxes
#     elif orig_format == BoxFormat.XC_YC_W_H:
#         new_boxes = np.copy(boxes)
#         new_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
#         new_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
#         new_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
#         new_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
#         return new_boxes
#     else:
#         raise Exception("Unsupported boxes format")
#
#
# def nms(dets, scores, iou_thres=0.5, max_out_dets=1000):
#     y1, x1 = dets[:, 0], dets[:, 1]
#     y2, x2 = dets[:, 2], dets[:, 3]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]
#
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#
#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#
#         inds = np.where(ovr <= iou_thres)[0]
#         order = order[inds + 1]
#
#     return keep[:max_out_dets]
#
#
# def _are_boxes_normalized(boxes):
#     if len(boxes) == 0:
#         return True  # it doesn't matter
#     if max(boxes[0]) > 1:
#         return False
#     return True
#
#
# def apply_normalization(boxes, orig_width, orig_height, boxes_format):
#     if _are_boxes_normalized(boxes):
#         return boxes
#     return _normalize_coordinates(boxes, orig_width, orig_height, boxes_format)
#
#
# def _normalize_coordinates(boxes, orig_width, orig_height, boxes_format):
#     """
#     gets boxes in the original images values and normalize them to be between 0 to 1
#
#     :param boxes:
#     :param orig_width: original image width
#     :param orig_height: original image height
#     :param boxes_format: if the boxes are in XMIN_YMIN_W_H or YMIM_XMIN_YMAX_XMAX format
#     :return:
#     """
#     if len(boxes) == 0:
#         return boxes
#     elif _are_boxes_normalized(boxes):
#         return boxes
#     convert_to_ymin_xmin_ymax_xmax_format(boxes, orig_format=boxes_format)
#     boxes[:, 0] = np.divide(boxes[:, 0], orig_height)
#     boxes[:, 1] = np.divide(boxes[:, 1], orig_width)
#     boxes[:, 2] = np.divide(boxes[:, 2], orig_height)
#     boxes[:, 3] = np.divide(boxes[:, 3], orig_width)
#     if boxes_format is BoxFormat.XMIN_YMIN_W_H:  # need to change back the boxes format to XMIN_YMIN_W_H
#         convert_to_xmin_ymin_width_height_format(boxes, BoxFormat.YMIM_XMIN_YMAX_XMAX)
#     return boxes
#
#
# class ObjectDetectionEvaluation(object):
#     """Internal implementation of Pascal object detection metrics."""
#
#     def __init__(self,
#                  num_groundtruth_classes,
#                  matching_iou_threshold=0.5,
#                  nms_iou_threshold=1.0,
#                  nms_max_output_boxes=10000,
#                  use_weighted_mean_ap=False,
#                  label_id_offset=0,
#                  group_of_weight=0.0):
#         if num_groundtruth_classes < 1:
#             raise ValueError('Need at least 1 groundtruth class for evaluation.')
#
#         self.per_image_eval = PerImageEvaluation(
#             num_groundtruth_classes=num_groundtruth_classes,
#             matching_iou_threshold=matching_iou_threshold,
#             nms_iou_threshold=nms_iou_threshold,
#             nms_max_output_boxes=nms_max_output_boxes,
#             group_of_weight=group_of_weight)
#         self.group_of_weight = group_of_weight
#         self.num_class = num_groundtruth_classes
#         self.use_weighted_mean_ap = use_weighted_mean_ap
#         self.label_id_offset = label_id_offset
#
#         self.groundtruth_boxes = {}
#         self.groundtruth_class_labels = {}
#         self.groundtruth_masks = {}
#         self.groundtruth_is_difficult_list = {}
#         self.groundtruth_is_group_of_list = {}
#         self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=float)
#         self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)
#
#         self._initialize_detections()
#
#     def _initialize_detections(self):
#         self.detection_keys = set()
#         self.scores_per_class = [[] for _ in range(self.num_class)]
#         self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
#         self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
#         self.average_precision_per_class = np.empty(self.num_class, dtype=float)
#         self.average_precision_per_class.fill(np.nan)
#         self.precisions_per_class = []
#         self.recalls_per_class = []
#         self.corloc_per_class = np.ones(self.num_class, dtype=float)
#
#     def clear_detections(self):
#         self._initialize_detections()
#
#     def add_single_ground_truth_image_info(self,
#                                            image_key,
#                                            groundtruth_boxes,
#                                            groundtruth_class_labels,
#                                            groundtruth_is_difficult_list=None,
#                                            groundtruth_is_group_of_list=None,
#                                            groundtruth_masks=None):
#         """Adds groundtruth for a single image to be used for evaluation.
#
#         Args:
#           image_key: A unique string/integer identifier for the image.
#           groundtruth_boxes: float32 numpy array of shape [num_boxes, 4]
#             containing `num_boxes` groundtruth boxes of the format
#             [ymin, xmin, ymax, xmax] in absolute image coordinates.
#           groundtruth_class_labels: integer numpy array of shape [num_boxes]
#             containing 0-indexed groundtruth classes for the boxes.
#           groundtruth_is_difficult_list: A length M numpy boolean array denoting
#             whether a ground truth box is a difficult instance or not. To support
#             the case that no boxes are difficult, it is by default set as None.
#           groundtruth_is_group_of_list: A length M numpy boolean array denoting
#               whether a ground truth box is a group-of box or not. To support
#               the case that no boxes are groups-of, it is by default set as None.
#           groundtruth_masks: uint8 numpy array of shape
#             [num_boxes, height, width] containing `num_boxes` groundtruth masks.
#             The mask values range from 0 to 1.
#         """
#         if image_key in self.groundtruth_boxes:
#             return
#
#         self.groundtruth_boxes[image_key] = groundtruth_boxes
#         self.groundtruth_class_labels[image_key] = groundtruth_class_labels
#         self.groundtruth_masks[image_key] = groundtruth_masks
#         if groundtruth_is_difficult_list is None:
#             num_boxes = groundtruth_boxes.shape[0]
#             groundtruth_is_difficult_list = np.zeros(num_boxes, dtype=bool)
#         self.groundtruth_is_difficult_list[
#             image_key] = groundtruth_is_difficult_list.astype(dtype=bool)
#         if groundtruth_is_group_of_list is None:
#             num_boxes = groundtruth_boxes.shape[0]
#             groundtruth_is_group_of_list = np.zeros(num_boxes, dtype=bool)
#         self.groundtruth_is_group_of_list[
#             image_key] = groundtruth_is_group_of_list.astype(dtype=bool)
#
#         self._update_ground_truth_statistics(
#             groundtruth_class_labels,
#             groundtruth_is_difficult_list.astype(dtype=bool),
#             groundtruth_is_group_of_list.astype(dtype=bool))
#
#     def add_single_detected_image_info(self, image_key, detected_boxes,
#                                        detected_scores, detected_class_labels,
#                                        detected_masks=None):
#         """Adds detections for a single image to be used for evaluation.
#
#         Args:
#           image_key: A unique string/integer identifier for the image.
#           detected_boxes: float32 numpy array of shape [num_boxes, 4]
#             containing `num_boxes` detection boxes of the format
#             [ymin, xmin, ymax, xmax] in absolute image coordinates.
#           detected_scores: float32 numpy array of shape [num_boxes] containing
#             detection scores for the boxes.
#           detected_class_labels: integer numpy array of shape [num_boxes] containing
#             0-indexed detection classes for the boxes.
#           detected_masks: np.uint8 numpy array of shape [num_boxes, height, width]
#             containing `num_boxes` detection masks with values ranging
#             between 0 and 1.
#
#         Raises:
#           ValueError: if the number of boxes, scores and class labels differ in
#             length.
#         """
#         if (len(detected_boxes) != len(detected_scores) or
#                 len(detected_boxes) != len(detected_class_labels)):
#             raise ValueError('detected_boxes, detected_scores and '
#                              'detected_class_labels should all have same lengths. Got'
#                              '[%d, %d, %d]' % len(detected_boxes),
#                              len(detected_scores), len(detected_class_labels))
#
#         if image_key in self.detection_keys:
#             return
#
#         self.detection_keys.add(image_key)
#         if image_key in self.groundtruth_boxes:
#             groundtruth_boxes = self.groundtruth_boxes[image_key]
#             groundtruth_class_labels = self.groundtruth_class_labels[image_key]
#             # Masks are popped instead of look up. The reason is that we do not want
#             # to keep all masks in memory which can cause memory overflow.
#             groundtruth_masks = self.groundtruth_masks.pop(
#                 image_key)
#             groundtruth_is_difficult_list = self.groundtruth_is_difficult_list[
#                 image_key]
#             groundtruth_is_group_of_list = self.groundtruth_is_group_of_list[
#                 image_key]
#         else:
#             groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
#             groundtruth_class_labels = np.array([], dtype=int)
#             if detected_masks is None:
#                 groundtruth_masks = None
#             else:
#                 groundtruth_masks = np.empty(shape=[0, 1, 1], dtype=float)
#             groundtruth_is_difficult_list = np.array([], dtype=bool)
#             groundtruth_is_group_of_list = np.array([], dtype=bool)
#         scores, tp_fp_labels, is_class_correctly_detected_in_image = (
#             self.per_image_eval.compute_object_detection_metrics(
#                 detected_boxes=detected_boxes,
#                 detected_scores=detected_scores,
#                 detected_class_labels=detected_class_labels,
#                 groundtruth_boxes=groundtruth_boxes,
#                 groundtruth_class_labels=groundtruth_class_labels,
#                 groundtruth_is_difficult_list=groundtruth_is_difficult_list,
#                 groundtruth_is_group_of_list=groundtruth_is_group_of_list,
#                 detected_masks=detected_masks,
#                 groundtruth_masks=groundtruth_masks))
#
#         for i in range(self.num_class):
#             if scores[i].shape[0] > 0:
#                 self.scores_per_class[i].append(scores[i])
#                 self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
#         self.num_images_correctly_detected_per_class += is_class_correctly_detected_in_image
#
#     def _update_ground_truth_statistics(self, groundtruth_class_labels,
#                                         groundtruth_is_difficult_list,
#                                         groundtruth_is_group_of_list):
#         """Update grouth truth statitistics.
#
#         1. Difficult boxes are ignored when counting the number of ground truth
#         instances as done in Pascal VOC devkit.
#         2. Difficult boxes are treated as normal boxes when computing CorLoc related
#         statitistics.
#
#         Args:
#           groundtruth_class_labels: An integer numpy array of length M,
#               representing M class labels of object instances in ground truth
#           groundtruth_is_difficult_list: A boolean numpy array of length M denoting
#               whether a ground truth box is a difficult instance or not
#           groundtruth_is_group_of_list: A boolean numpy array of length M denoting
#               whether a ground truth box is a group-of box or not
#         """
#         for class_index in range(self.num_class):
#             num_gt_instances = np.sum(groundtruth_class_labels[
#                                           ~groundtruth_is_difficult_list
#                                           & ~groundtruth_is_group_of_list] == class_index)
#             num_groupof_gt_instances = self.group_of_weight * np.sum(
#                 groundtruth_class_labels[groundtruth_is_group_of_list] == class_index)
#             self.num_gt_instances_per_class[
#                 class_index] += num_gt_instances + num_groupof_gt_instances
#             if np.any(groundtruth_class_labels == class_index):
#                 self.num_gt_imgs_per_class[class_index] += 1
#
#     def evaluate(self):
#         """Compute evaluation result.
#
#         Returns:
#           A named tuple with the following fields -
#             average_precision: float numpy array of average precision for
#                 each class.
#             mean_ap: mean average precision of all classes, float scalar
#             precisions: List of precisions, each precision is a float numpy
#                 array
#             recalls: List of recalls, each recall is a float numpy array
#             corloc: numpy float array
#             mean_corloc: Mean CorLoc score for each class, float scalar
#         """
#         if False:  # (self.num_gt_instances_per_class == 0).any():
#             logging.warn(
#                 'The following classes have no ground truth examples: %s',
#                 np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0)) +
#                 self.label_id_offset)
#
#         if self.use_weighted_mean_ap:
#             all_scores = np.array([], dtype=float)
#             all_tp_fp_labels = np.array([], dtype=bool)
#         for class_index in range(self.num_class):
#             if self.num_gt_instances_per_class[class_index] == 0:
#                 continue
#             if not self.scores_per_class[class_index]:
#                 scores = np.array([], dtype=float)
#                 tp_fp_labels = np.array([], dtype=float)
#             else:
#                 scores = np.concatenate(self.scores_per_class[class_index])
#                 tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
#             if self.use_weighted_mean_ap:
#                 all_scores = np.append(all_scores, scores)
#                 all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
#             precision, recall = metrics.compute_precision_recall(
#                 scores, tp_fp_labels, self.num_gt_instances_per_class[class_index])
#             self.precisions_per_class.append(precision)
#             self.recalls_per_class.append(recall)
#             average_precision = metrics.compute_average_precision(precision, recall)
#             self.average_precision_per_class[class_index] = average_precision
#
#         self.corloc_per_class = metrics.compute_cor_loc(
#             self.num_gt_imgs_per_class,
#             self.num_images_correctly_detected_per_class)
#
#         if self.use_weighted_mean_ap:
#             num_gt_instances = np.sum(self.num_gt_instances_per_class)
#             precision, recall = metrics.compute_precision_recall(
#                 all_scores, all_tp_fp_labels, num_gt_instances)
#             mean_ap = metrics.compute_average_precision(precision, recall)
#         else:
#             mean_ap = np.nanmean(self.average_precision_per_class)
#         mean_corloc = np.nanmean(self.corloc_per_class)
#         return ObjectDetectionEvalMetrics(
#             self.average_precision_per_class, mean_ap, self.precisions_per_class,
#             self.recalls_per_class, self.corloc_per_class, mean_corloc)
#
#
# class PerImageEvaluation(object):
#     """Evaluate detection result of a single image."""
#
#     def __init__(self,
#                  num_groundtruth_classes,
#                  matching_iou_threshold=0.5,
#                  nms_iou_threshold=0.3,
#                  nms_max_output_boxes=50,
#                  group_of_weight=0.0):
#         """Initialized PerImageEvaluation by evaluation parameters.
#
#     Args:
#       num_groundtruth_classes: Number of ground truth object classes
#       matching_iou_threshold: A ratio of area intersection to union, which is
#           the threshold to consider whether a detection is true positive or not
#       nms_iou_threshold: IOU threshold used in Non Maximum Suppression.
#       nms_max_output_boxes: Number of maximum output boxes in NMS.
#       group_of_weight: Weight of the group-of boxes.
#     """
#         self.matching_iou_threshold = matching_iou_threshold
#         self.nms_iou_threshold = nms_iou_threshold
#         self.nms_max_output_boxes = nms_max_output_boxes
#         self.num_groundtruth_classes = num_groundtruth_classes
#         self.group_of_weight = group_of_weight
#
#     def compute_object_detection_metrics(
#             self, detected_boxes, detected_scores, detected_class_labels,
#             groundtruth_boxes, groundtruth_class_labels,
#             groundtruth_is_difficult_list, groundtruth_is_group_of_list,
#             detected_masks=None, groundtruth_masks=None):
#         """Evaluates detections as being tp, fp or weighted from a single image.
#
#     The evaluation is done in two stages:
#      1. All detections are matched to non group-of boxes; true positives are
#         determined and detections matched to difficult boxes are ignored.
#      2. Detections that are determined as false positives are matched against
#         group-of boxes and weighted if matched.
#
#     Args:
#       detected_boxes: A float numpy array of shape [N, 4], representing N
#           regions of detected object regions.
#           Each row is of the format [y_min, x_min, y_max, x_max]
#       detected_scores: A float numpy array of shape [N, 1], representing
#           the confidence scores of the detected N object instances.
#       detected_class_labels: A integer numpy array of shape [N, 1], repreneting
#           the class labels of the detected N object instances.
#       groundtruth_boxes: A float numpy array of shape [M, 4], representing M
#           regions of object instances in ground truth
#       groundtruth_class_labels: An integer numpy array of shape [M, 1],
#           representing M class labels of object instances in ground truth
#       groundtruth_is_difficult_list: A boolean numpy array of length M denoting
#           whether a ground truth box is a difficult instance or not
#       groundtruth_is_group_of_list: A boolean numpy array of length M denoting
#           whether a ground truth box has group-of tag
#       detected_masks: (optional) A uint8 numpy array of shape
#         [N, height, width]. If not None, the metrics will be computed based
#         on masks.
#       groundtruth_masks: (optional) A uint8 numpy array of shape
#         [M, height, width].
#
#     Returns:
#       scores: A list of C float numpy arrays. Each numpy array is of
#           shape [K, 1], representing K scores detected with object class
#           label c
#       tp_fp_labels: A list of C boolean numpy arrays. Each numpy array
#           is of shape [K, 1], representing K True/False positive label of
#           object instances detected with class label c
#       is_class_correctly_detected_in_image: a numpy integer array of
#           shape [C, 1], indicating whether the correponding class has a least
#           one instance being correctly detected in the image
#     """
#         detected_boxes, detected_scores, detected_class_labels, detected_masks = (
#             self._remove_invalid_boxes(detected_boxes, detected_scores,
#                                        detected_class_labels, detected_masks))
#         scores, tp_fp_labels = self._compute_tp_fp(
#             detected_boxes=detected_boxes,
#             detected_scores=detected_scores,
#             detected_class_labels=detected_class_labels,
#             groundtruth_boxes=groundtruth_boxes,
#             groundtruth_class_labels=groundtruth_class_labels,
#             groundtruth_is_difficult_list=groundtruth_is_difficult_list,
#             groundtruth_is_group_of_list=groundtruth_is_group_of_list,
#             detected_masks=detected_masks,
#             groundtruth_masks=groundtruth_masks)
#
#         is_class_correctly_detected_in_image = self._compute_cor_loc(
#             detected_boxes=detected_boxes,
#             detected_scores=detected_scores,
#             detected_class_labels=detected_class_labels,
#             groundtruth_boxes=groundtruth_boxes,
#             groundtruth_class_labels=groundtruth_class_labels,
#             detected_masks=detected_masks,
#             groundtruth_masks=groundtruth_masks)
#
#         return scores, tp_fp_labels, is_class_correctly_detected_in_image
#
#     def _compute_cor_loc(self, detected_boxes, detected_scores,
#                          detected_class_labels, groundtruth_boxes,
#                          groundtruth_class_labels, detected_masks=None,
#                          groundtruth_masks=None):
#         """Compute CorLoc score for object detection result.
#
#     Args:
#       detected_boxes: A float numpy array of shape [N, 4], representing N
#           regions of detected object regions.
#           Each row is of the format [y_min, x_min, y_max, x_max]
#       detected_scores: A float numpy array of shape [N, 1], representing
#           the confidence scores of the detected N object instances.
#       detected_class_labels: A integer numpy array of shape [N, 1], repreneting
#           the class labels of the detected N object instances.
#       groundtruth_boxes: A float numpy array of shape [M, 4], representing M
#           regions of object instances in ground truth
#       groundtruth_class_labels: An integer numpy array of shape [M, 1],
#           representing M class labels of object instances in ground truth
#       detected_masks: (optional) A uint8 numpy array of shape
#         [N, height, width]. If not None, the scores will be computed based
#         on masks.
#       groundtruth_masks: (optional) A uint8 numpy array of shape
#         [M, height, width].
#
#     Returns:
#       is_class_correctly_detected_in_image: a numpy integer array of
#           shape [C, 1], indicating whether the correponding class has a least
#           one instance being correctly detected in the image
#
#     Raises:
#       ValueError: If detected masks is not None but groundtruth masks are None,
#         or the other way around.
#     """
#         if (detected_masks is not None and
#             groundtruth_masks is None) or (detected_masks is None and
#                                            groundtruth_masks is not None):
#             raise ValueError(
#                 'If `detected_masks` is provided, then `groundtruth_masks` should '
#                 'also be provided.'
#             )
#
#         is_class_correctly_detected_in_image = np.zeros(
#             self.num_groundtruth_classes, dtype=int)
#         for i in range(self.num_groundtruth_classes):
#             (gt_boxes_at_ith_class, gt_masks_at_ith_class,
#              detected_boxes_at_ith_class, detected_scores_at_ith_class,
#              detected_masks_at_ith_class) = self._get_ith_class_arrays(
#                 detected_boxes, detected_scores, detected_masks,
#                 detected_class_labels, groundtruth_boxes, groundtruth_masks,
#                 groundtruth_class_labels, i)
#             is_class_correctly_detected_in_image[i] = (
#                 self._compute_is_class_correctly_detected_in_image(
#                     detected_boxes=detected_boxes_at_ith_class,
#                     detected_scores=detected_scores_at_ith_class,
#                     groundtruth_boxes=gt_boxes_at_ith_class,
#                     detected_masks=detected_masks_at_ith_class,
#                     groundtruth_masks=gt_masks_at_ith_class))
#
#         return is_class_correctly_detected_in_image
#
#     def _compute_is_class_correctly_detected_in_image(
#             self, detected_boxes, detected_scores, groundtruth_boxes,
#             detected_masks=None, groundtruth_masks=None):
#         """Compute CorLoc score for a single class.
#
#     Args:
#       detected_boxes: A numpy array of shape [N, 4] representing detected box
#           coordinates
#       detected_scores: A 1-d numpy array of length N representing classification
#           score
#       groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
#           box coordinates
#       detected_masks: (optional) A np.uint8 numpy array of shape
#         [N, height, width]. If not None, the scores will be computed based
#         on masks.
#       groundtruth_masks: (optional) A np.uint8 numpy array of shape
#         [M, height, width].
#
#     Returns:
#       is_class_correctly_detected_in_image: An integer 1 or 0 denoting whether a
#           class is correctly detected in the image or not
#     """
#         if detected_boxes.size > 0:
#             if groundtruth_boxes.size > 0:
#                 max_score_id = np.argmax(detected_scores)
#                 mask_mode = False
#                 if detected_masks is not None and groundtruth_masks is not None:
#                     mask_mode = True
#                 if mask_mode:
#                     detected_boxlist = BoxMaskList(
#                         box_data=np.expand_dims(detected_boxes[max_score_id], axis=0),
#                         mask_data=np.expand_dims(detected_masks[max_score_id], axis=0))
#                     gt_boxlist = BoxMaskList(
#                         box_data=groundtruth_boxes, mask_data=groundtruth_masks)
#                     iou = np_box_mask_list_ops.iou(detected_boxlist, gt_boxlist)
#                 else:
#                     detected_boxlist = BoxList(
#                         np.expand_dims(detected_boxes[max_score_id, :], axis=0))
#                     gt_boxlist = BoxList(groundtruth_boxes)
#                     iou = np_box_list_ops.iou(detected_boxlist, gt_boxlist)
#                 if np.max(iou) >= self.matching_iou_threshold:
#                     return 1
#         return 0
#
#     def _compute_tp_fp(self, detected_boxes, detected_scores,
#                        detected_class_labels, groundtruth_boxes,
#                        groundtruth_class_labels, groundtruth_is_difficult_list,
#                        groundtruth_is_group_of_list,
#                        detected_masks=None, groundtruth_masks=None):
#         """Labels true/false positives of detections of an image across all classes.
#
#     Args:
#       detected_boxes: A float numpy array of shape [N, 4], representing N
#           regions of detected object regions.
#           Each row is of the format [y_min, x_min, y_max, x_max]
#       detected_scores: A float numpy array of shape [N, 1], representing
#           the confidence scores of the detected N object instances.
#       detected_class_labels: A integer numpy array of shape [N, 1], repreneting
#           the class labels of the detected N object instances.
#       groundtruth_boxes: A float numpy array of shape [M, 4], representing M
#           regions of object instances in ground truth
#       groundtruth_class_labels: An integer numpy array of shape [M, 1],
#           representing M class labels of object instances in ground truth
#       groundtruth_is_difficult_list: A boolean numpy array of length M denoting
#           whether a ground truth box is a difficult instance or not
#       groundtruth_is_group_of_list: A boolean numpy array of length M denoting
#           whether a ground truth box has group-of tag
#       detected_masks: (optional) A np.uint8 numpy array of shape
#         [N, height, width]. If not None, the scores will be computed based
#         on masks.
#       groundtruth_masks: (optional) A np.uint8 numpy array of shape
#         [M, height, width].
#
#     Returns:
#       result_scores: A list of float numpy arrays. Each numpy array is of
#           shape [K, 1], representing K scores detected with object class
#           label c
#       result_tp_fp_labels: A list of boolean numpy array. Each numpy array is of
#           shape [K, 1], representing K True/False positive label of object
#           instances detected with class label c
#
#     Raises:
#       ValueError: If detected masks is not None but groundtruth masks are None,
#         or the other way around.
#     """
#         if detected_masks is not None and groundtruth_masks is None:
#             raise ValueError(
#                 'Detected masks is available but groundtruth masks is not.')
#         if detected_masks is None and groundtruth_masks is not None:
#             raise ValueError(
#                 'Groundtruth masks is available but detected masks is not.')
#
#         result_scores = []
#         result_tp_fp_labels = []
#         for i in range(self.num_groundtruth_classes):
#             groundtruth_is_difficult_list_at_ith_class = (
#                 groundtruth_is_difficult_list[groundtruth_class_labels == i])
#             groundtruth_is_group_of_list_at_ith_class = (
#                 groundtruth_is_group_of_list[groundtruth_class_labels == i])
#             (gt_boxes_at_ith_class, gt_masks_at_ith_class,
#              detected_boxes_at_ith_class, detected_scores_at_ith_class,
#              detected_masks_at_ith_class) = self._get_ith_class_arrays(
#                 detected_boxes, detected_scores, detected_masks,
#                 detected_class_labels, groundtruth_boxes, groundtruth_masks,
#                 groundtruth_class_labels, i)
#             scores, tp_fp_labels = self._compute_tp_fp_for_single_class(
#                 detected_boxes=detected_boxes_at_ith_class,
#                 detected_scores=detected_scores_at_ith_class,
#                 groundtruth_boxes=gt_boxes_at_ith_class,
#                 groundtruth_is_difficult_list=
#                 groundtruth_is_difficult_list_at_ith_class,
#                 groundtruth_is_group_of_list=
#                 groundtruth_is_group_of_list_at_ith_class,
#                 detected_masks=detected_masks_at_ith_class,
#                 groundtruth_masks=gt_masks_at_ith_class)
#             result_scores.append(scores)
#             result_tp_fp_labels.append(tp_fp_labels)
#         return result_scores, result_tp_fp_labels
#
#     def _get_overlaps_and_scores_mask_mode(
#             self, detected_boxes, detected_scores, detected_masks, groundtruth_boxes,
#             groundtruth_masks, groundtruth_is_group_of_list):
#         """Computes overlaps and scores between detected and groudntruth masks.
#
#     Args:
#       detected_boxes: A numpy array of shape [N, 4] representing detected box
#           coordinates
#       detected_scores: A 1-d numpy array of length N representing classification
#           score
#       detected_masks: A uint8 numpy array of shape [N, height, width]. If not
#           None, the scores will be computed based on masks.
#       groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
#           box coordinates
#       groundtruth_masks: A uint8 numpy array of shape [M, height, width].
#       groundtruth_is_group_of_list: A boolean numpy array of length M denoting
#           whether a ground truth box has group-of tag. If a groundtruth box
#           is group-of box, every detection matching this box is ignored.
#
#     Returns:
#       iou: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
#           gt_non_group_of_boxlist.num_boxes() == 0 it will be None.
#       ioa: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
#           gt_group_of_boxlist.num_boxes() == 0 it will be None.
#       scores: The score of the detected boxlist.
#       num_boxes: Number of non-maximum suppressed detected boxes.
#     """
#         detected_boxlist = np_box_mask_list.BoxMaskList(
#             box_data=detected_boxes, mask_data=detected_masks)
#         detected_boxlist.add_field('scores', detected_scores)
#         detected_boxlist = non_max_suppression(
#             detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)
#         gt_non_group_of_boxlist = BoxMaskList(
#             box_data=groundtruth_boxes[~groundtruth_is_group_of_list],
#             mask_data=groundtruth_masks[~groundtruth_is_group_of_list])
#         gt_group_of_boxlist = BoxMaskList(
#             box_data=groundtruth_boxes[groundtruth_is_group_of_list],
#             mask_data=groundtruth_masks[groundtruth_is_group_of_list])
#         iou = iou(detected_boxlist, gt_non_group_of_boxlist)
#         ioa = np.transpose(
#             ioa(gt_group_of_boxlist, detected_boxlist))
#         scores = detected_boxlist.get_field('scores')
#         num_boxes = detected_boxlist.num_boxes()
#         return iou, ioa, scores, num_boxes
#
#     def _get_overlaps_and_scores_box_mode(
#             self,
#             detected_boxes,
#             detected_scores,
#             groundtruth_boxes,
#             groundtruth_is_group_of_list):
#         """Computes overlaps and scores between detected and groudntruth boxes.
#
#     Args:
#       detected_boxes: A numpy array of shape [N, 4] representing detected box
#           coordinates
#       detected_scores: A 1-d numpy array of length N representing classification
#           score
#       groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
#           box coordinates
#       groundtruth_is_group_of_list: A boolean numpy array of length M denoting
#           whether a ground truth box has group-of tag. If a groundtruth box
#           is group-of box, every detection matching this box is ignored.
#
#     Returns:
#       iou: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
#           gt_non_group_of_boxlist.num_boxes() == 0 it will be None.
#       ioa: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
#           gt_group_of_boxlist.num_boxes() == 0 it will be None.
#       scores: The score of the detected boxlist.
#       num_boxes: Number of non-maximum suppressed detected boxes.
#     """
#         detected_boxlist = BoxList(detected_boxes)
#         detected_boxlist.add_field('scores', detected_scores)
#         detected_boxlist = non_max_suppression(
#             detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)
#         gt_non_group_of_boxlist = BoxList(
#             groundtruth_boxes[~groundtruth_is_group_of_list])
#         gt_group_of_boxlist = BoxList(
#             groundtruth_boxes[groundtruth_is_group_of_list])
#         iou = iou(detected_boxlist, gt_non_group_of_boxlist)
#         ioa = np.transpose(
#             ioa(gt_group_of_boxlist, detected_boxlist))
#         scores = detected_boxlist.get_field('scores')
#         num_boxes = detected_boxlist.num_boxes()
#         return iou, ioa, scores, num_boxes
#
#     def _compute_tp_fp_for_single_class(
#             self, detected_boxes, detected_scores, groundtruth_boxes,
#             groundtruth_is_difficult_list, groundtruth_is_group_of_list,
#             detected_masks=None, groundtruth_masks=None):
#         """Labels boxes detected with the same class from the same image as tp/fp.
#
#     Args:
#       detected_boxes: A numpy array of shape [N, 4] representing detected box
#           coordinates
#       detected_scores: A 1-d numpy array of length N representing classification
#           score
#       groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
#           box coordinates
#       groundtruth_is_difficult_list: A boolean numpy array of length M denoting
#           whether a ground truth box is a difficult instance or not. If a
#           groundtruth box is difficult, every detection matching this box
#           is ignored.
#       groundtruth_is_group_of_list: A boolean numpy array of length M denoting
#           whether a ground truth box has group-of tag. If a groundtruth box
#           is group-of box, every detection matching this box is ignored.
#       detected_masks: (optional) A uint8 numpy array of shape
#         [N, height, width]. If not None, the scores will be computed based
#         on masks.
#       groundtruth_masks: (optional) A uint8 numpy array of shape
#         [M, height, width].
#
#     Returns:
#       Two arrays of the same size, containing all boxes that were evaluated as
#       being true positives or false positives; if a box matched to a difficult
#       box or to a group-of box, it is ignored.
#
#       scores: A numpy array representing the detection scores.
#       tp_fp_labels: a boolean numpy array indicating whether a detection is a
#           true positive.
#     """
#         if detected_boxes.size == 0:
#             return np.array([], dtype=float), np.array([], dtype=bool)
#
#         mask_mode = False
#         if detected_masks is not None and groundtruth_masks is not None:
#             mask_mode = True
#
#         if mask_mode:
#             (iou, ioa, scores,
#              num_detected_boxes) = self._get_overlaps_and_scores_mask_mode(
#                 detected_boxes=detected_boxes,
#                 detected_scores=detected_scores,
#                 detected_masks=detected_masks,
#                 groundtruth_boxes=groundtruth_boxes,
#                 groundtruth_masks=groundtruth_masks,
#                 groundtruth_is_group_of_list=groundtruth_is_group_of_list)
#         else:
#             (iou, ioa, scores,
#              num_detected_boxes) = self._get_overlaps_and_scores_box_mode(
#                 detected_boxes=detected_boxes,
#                 detected_scores=detected_scores,
#                 groundtruth_boxes=groundtruth_boxes,
#                 groundtruth_is_group_of_list=groundtruth_is_group_of_list)
#
#         if groundtruth_boxes.size == 0:
#             return scores, np.zeros(num_detected_boxes, dtype=bool)
#
#         tp_fp_labels = np.zeros(num_detected_boxes, dtype=bool)
#         is_matched_to_difficult_box = np.zeros(num_detected_boxes, dtype=bool)
#         is_matched_to_group_of_box = np.zeros(num_detected_boxes, dtype=bool)
#
#         # The evaluation is done in two stages:
#         # 1. All detections are matched to non group-of boxes; true positives are
#         #    determined and detections matched to difficult boxes are ignored.
#         # 2. Detections that are determined as false positives are matched against
#         #    group-of boxes and scored with weight w per ground truth box is
#         # matched.
#
#         # Tp-fp evaluation for non-group of boxes (if any).
#         if iou.shape[1] > 0:
#             groundtruth_nongroup_of_is_difficult_list = groundtruth_is_difficult_list[
#                 ~groundtruth_is_group_of_list]
#             max_overlap_gt_ids = np.argmax(iou, axis=1)
#             is_gt_box_detected = np.zeros(iou.shape[1], dtype=bool)
#             for i in range(num_detected_boxes):
#                 gt_id = max_overlap_gt_ids[i]
#                 if iou[i, gt_id] >= self.matching_iou_threshold:
#                     # if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
#                     if not is_gt_box_detected[gt_id]:
#                         tp_fp_labels[i] = True
#                         is_gt_box_detected[gt_id] = True
#                     else:
#                         is_matched_to_difficult_box[i] = True
#
#         scores_group_of = np.zeros(ioa.shape[1], dtype=float)
#         tp_fp_labels_group_of = self.group_of_weight * np.ones(
#             ioa.shape[1], dtype=float)
#         # Tp-fp evaluation for group of boxes.
#         if ioa.shape[1] > 0:
#             max_overlap_group_of_gt_ids = np.argmax(ioa, axis=1)
#             for i in range(num_detected_boxes):
#                 gt_id = max_overlap_group_of_gt_ids[i]
#                 if (not tp_fp_labels[i] and not is_matched_to_difficult_box[i] and
#                         ioa[i, gt_id] >= self.matching_iou_threshold):
#                     is_matched_to_group_of_box[i] = True
#                     scores_group_of[gt_id] = max(scores_group_of[gt_id], scores[i])
#             selector = np.where((scores_group_of > 0) & (tp_fp_labels_group_of > 0))
#             scores_group_of = scores_group_of[selector]
#             tp_fp_labels_group_of = tp_fp_labels_group_of[selector]
#
#         return np.concatenate(
#             (scores[~is_matched_to_difficult_box
#                     & ~is_matched_to_group_of_box],
#              scores_group_of)), np.concatenate(
#             (tp_fp_labels[~is_matched_to_difficult_box
#                           & ~is_matched_to_group_of_box].astype(float),
#              tp_fp_labels_group_of))
#
#     def _get_ith_class_arrays(self, detected_boxes, detected_scores,
#                               detected_masks, detected_class_labels,
#                               groundtruth_boxes, groundtruth_masks,
#                               groundtruth_class_labels, class_index):
#         """Returns numpy arrays belonging to class with index `class_index`.
#
#     Args:
#       detected_boxes: A numpy array containing detected boxes.
#       detected_scores: A numpy array containing detected scores.
#       detected_masks: A numpy array containing detected masks.
#       detected_class_labels: A numpy array containing detected class labels.
#       groundtruth_boxes: A numpy array containing groundtruth boxes.
#       groundtruth_masks: A numpy array containing groundtruth masks.
#       groundtruth_class_labels: A numpy array containing groundtruth class
#         labels.
#       class_index: An integer index.
#
#     Returns:
#       gt_boxes_at_ith_class: A numpy array containing groundtruth boxes labeled
#         as ith class.
#       gt_masks_at_ith_class: A numpy array containing groundtruth masks labeled
#         as ith class.
#       detected_boxes_at_ith_class: A numpy array containing detected boxes
#         corresponding to the ith class.
#       detected_scores_at_ith_class: A numpy array containing detected scores
#         corresponding to the ith class.
#       detected_masks_at_ith_class: A numpy array containing detected masks
#         corresponding to the ith class.
#     """
#         selected_groundtruth = (groundtruth_class_labels == class_index)
#         gt_boxes_at_ith_class = groundtruth_boxes[selected_groundtruth]
#         if groundtruth_masks is not None:
#             gt_masks_at_ith_class = groundtruth_masks[selected_groundtruth]
#         else:
#             gt_masks_at_ith_class = None
#         selected_detections = (detected_class_labels == class_index)
#         detected_boxes_at_ith_class = detected_boxes[selected_detections]
#         detected_scores_at_ith_class = detected_scores[selected_detections]
#         if detected_masks is not None:
#             detected_masks_at_ith_class = detected_masks[selected_detections]
#         else:
#             detected_masks_at_ith_class = None
#         return (gt_boxes_at_ith_class, gt_masks_at_ith_class,
#                 detected_boxes_at_ith_class, detected_scores_at_ith_class,
#                 detected_masks_at_ith_class)
#
#     def _remove_invalid_boxes(self, detected_boxes, detected_scores,
#                               detected_class_labels, detected_masks=None):
#         """Removes entries with invalid boxes.
#
#     A box is invalid if either its xmax is smaller than its xmin, or its ymax
#     is smaller than its ymin.
#
#     Args:
#       detected_boxes: A float numpy array of size [num_boxes, 4] containing box
#         coordinates in [ymin, xmin, ymax, xmax] format.
#       detected_scores: A float numpy array of size [num_boxes].
#       detected_class_labels: A int32 numpy array of size [num_boxes].
#       detected_masks: A uint8 numpy array of size [num_boxes, height, width].
#
#     Returns:
#       valid_detected_boxes: A float numpy array of size [num_valid_boxes, 4]
#         containing box coordinates in [ymin, xmin, ymax, xmax] format.
#       valid_detected_scores: A float numpy array of size [num_valid_boxes].
#       valid_detected_class_labels: A int32 numpy array of size
#         [num_valid_boxes].
#       valid_detected_masks: A uint8 numpy array of size
#         [num_valid_boxes, height, width].
#     """
#         valid_indices = np.logical_and(detected_boxes[:, 0] < detected_boxes[:, 2],
#                                        detected_boxes[:, 1] < detected_boxes[:, 3])
#         detected_boxes = detected_boxes[valid_indices]
#         detected_scores = detected_scores[valid_indices]
#         detected_class_labels = detected_class_labels[valid_indices]
#         if detected_masks is not None:
#             detected_masks = detected_masks[valid_indices]
#         return [
#             detected_boxes, detected_scores, detected_class_labels, detected_masks
#         ]
#
# #
# # class BoxMaskList(BoxList):
# #     """Convenience wrapper for BoxList with masks.
# #
# #   BoxMaskList extends the np_box_list.BoxList to contain masks as well.
# #   In particular, its constructor receives both boxes and masks. Note that the
# #   masks correspond to the full image.
# #   """
# #
# #     def __init__(self, box_data, mask_data):
# #         """Constructs box collection.
# #
# #     Args:
# #       box_data: a numpy array of shape [N, 4] representing box coordinates
# #       mask_data: a numpy array of shape [N, height, width] representing masks
# #         with values are in {0,1}. The masks correspond to the full
# #         image. The height and the width will be equal to image height and width.
# #
# #     Raises:
# #       ValueError: if bbox data is not a numpy array
# #       ValueError: if invalid dimensions for bbox data
# #       ValueError: if mask data is not a numpy array
# #       ValueError: if invalid dimension for mask data
# #     """
# #         super(BoxMaskList, self).__init__(box_data)
# #         if not isinstance(mask_data, np.ndarray):
# #             raise ValueError('Mask data must be a numpy array.')
# #         if len(mask_data.shape) != 3:
# #             raise ValueError('Invalid dimensions for mask data.')
# #         if mask_data.dtype != np.uint8:
# #             raise ValueError('Invalid data type for mask data: uint8 is required.')
# #         if mask_data.shape[0] != box_data.shape[0]:
# #             raise ValueError('There should be the same number of boxes and masks.')
# #         self.data['masks'] = mask_data
# #
# #     def get_masks(self):
# #         """Convenience function for accessing masks.
# #
# #     Returns:
# #       a numpy array of shape [N, height, width] representing masks
# #     """
# #         return self.get_field('masks')
