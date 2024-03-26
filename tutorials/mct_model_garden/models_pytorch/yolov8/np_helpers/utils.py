"""
This module contains functions that changes the box format
"""
from enum import Enum
import numpy as np


class BoxFormat(Enum):
    YMIM_XMIN_YMAX_XMAX = 'ymin_xmin_ymax_xmax'
    XMIM_YMIN_XMAX_YMAX = 'xmin_ymin_xmax_ymax'
    XMIN_YMIN_W_H = 'xmin_ymin_width_height'
    XC_YC_W_H = 'xc_yc_width_height'

    def __eq__(self, other):
        return self.value == other.value

def convert_to_xmin_ymin_width_height_format(boxes, orig_format: BoxFormat):
    """
    changes the box from one format to another (YMIM_XMIN_YMAX_XMAX --> XMIN_YMIN_W_H)
    also support in same format mode (returns the same format)

    :param boxes:
    :param orig_format:
    :return: box in format XMIN_YMIN_W_H
    """
    if len(boxes) == 0:
        return boxes
    elif orig_format == BoxFormat.XMIN_YMIN_W_H:
        return boxes
    elif orig_format == BoxFormat.YMIM_XMIN_YMAX_XMAX:
        boxes[:, 2] -= boxes[:, 0]  # convert ymax to height
        boxes[:, 3] -= boxes[:, 1]  # convert xmax to width
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap ymin, xmin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap height, width columns
        return boxes
    elif orig_format == BoxFormat.XMIM_YMIN_XMAX_YMAX:
        boxes[:, 2] -= boxes[:, 0]  # convert ymax to height
        boxes[:, 3] -= boxes[:, 1]  # convert xmax to width
        return boxes
    elif orig_format == BoxFormat.XC_YC_W_H:
        new_boxes = np.copy(boxes)
        new_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
        new_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
        new_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
        new_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
        return new_boxes
    else:
        raise Exception("Unsupported boxes format")


def convert_to_ymin_xmin_ymax_xmax_format(boxes, orig_format: BoxFormat):
    """
    changes the box from one format to another (XMIN_YMIN_W_H --> YMIM_XMIN_YMAX_XMAX )
    also support in same format mode (returns the same format)

    :param boxes:
    :param orig_format:
    :return: box in format YMIM_XMIN_YMAX_XMAX
    """
    if len(boxes) == 0:
        return boxes
    elif orig_format == BoxFormat.YMIM_XMIN_YMAX_XMAX:
        return boxes
    elif orig_format == BoxFormat.XMIN_YMIN_W_H:
        boxes[:, 2] += boxes[:, 0]  # convert width to xmax
        boxes[:, 3] += boxes[:, 1]  # convert height to ymax
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XMIM_YMIN_XMAX_YMAX:
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XC_YC_W_H:
        new_boxes = np.copy(boxes)
        new_boxes[:, 0] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
        new_boxes[:, 1] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
        new_boxes[:, 2] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
        new_boxes[:, 3] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
        return new_boxes
    else:
        raise Exception("Unsupported boxes format")


def _normalize_coordinates(boxes, orig_width, orig_height, boxes_format):
    """
    gets boxes in the original images values and normalize them to be between 0 to 1

    :param boxes:
    :param orig_width: original image width
    :param orig_height: original image height
    :param boxes_format: if the boxes are in XMIN_YMIN_W_H or YMIM_XMIN_YMAX_XMAX format
    :return:
    """
    if len(boxes) == 0:
        return boxes
    elif _are_boxes_normalized(boxes):
        return boxes
    convert_to_ymin_xmin_ymax_xmax_format(boxes, orig_format=boxes_format)
    boxes[:, 0] = np.divide(boxes[:, 0], orig_height)
    boxes[:, 1] = np.divide(boxes[:, 1], orig_width)
    boxes[:, 2] = np.divide(boxes[:, 2], orig_height)
    boxes[:, 3] = np.divide(boxes[:, 3], orig_width)
    if boxes_format is BoxFormat.XMIN_YMIN_W_H:  # need to change back the boxes format to XMIN_YMIN_W_H
        convert_to_xmin_ymin_width_height_format(boxes, BoxFormat.YMIM_XMIN_YMAX_XMAX)
    return boxes


def _denormalize_coordinates(boxes, orig_width, orig_height, boxes_format):
    """
    gets boxes normalized between 0 to 1 and in denormalized them to be in the original images values

    :param boxes:
    :param orig_width: original image width
    :param orig_height: original image height
    :param boxes_format: if the boxes are in XMIN_YMIN_W_H or YMIM_XMIN_YMAX_XMAX format
    :return:
    """
    if len(boxes) == 0:
        return boxes
    elif not _are_boxes_normalized(boxes):
        return boxes
    convert_to_ymin_xmin_ymax_xmax_format(boxes, orig_format=boxes_format)
    boxes[:, 0] = np.multiply(boxes[:, 0], orig_height)
    boxes[:, 1] = np.multiply(boxes[:, 1], orig_width)
    boxes[:, 2] = np.multiply(boxes[:, 2], orig_height)
    boxes[:, 3] = np.multiply(boxes[:, 3], orig_width)
    if boxes_format is BoxFormat.XMIN_YMIN_W_H:  # need to change back the boxes format to XMIN_YMIN_W_H
        convert_to_xmin_ymin_width_height_format(boxes, BoxFormat.YMIM_XMIN_YMAX_XMAX)
    return boxes


def are_ann_normalized(annotations):
    """
    returns whether the boxes in annotations are normalized between 0-1

    :param annotations: boxes and scores
    :return:
    """
    gt_boxes = annotations[0]['boxes']
    for ann in annotations:
        if len(ann['boxes']) > 0:
            return _are_boxes_normalized(gt_boxes)


def _are_boxes_normalized(boxes):
    if len(boxes) == 0:
        return True  # it doesn't matter
    if max(boxes[0]) > 1:
        return False
    return True


def apply_normalization(boxes, orig_width, orig_height, boxes_format):
    if _are_boxes_normalized(boxes):
        return boxes
    return _normalize_coordinates(boxes, orig_width, orig_height, boxes_format)


def apply_denormalization(boxes, orig_width, orig_height, boxes_format):
    if _are_boxes_normalized(boxes):
        return _denormalize_coordinates(boxes, orig_width, orig_height, boxes_format)
    return boxes

