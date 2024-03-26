import numpy as np
import pickle

from tutorials.mct_model_garden.models_pytorch.yolov8.np_helpers.metrics import compute_average_precision, \
    compute_cor_loc, compute_precision_recall
from tutorials.mct_model_garden.models_pytorch.yolov8.np_helpers.per_image_evaluation import PerImageEvaluation
from tutorials.mct_model_garden.models_pytorch.yolov8.np_helpers.utils import convert_to_ymin_xmin_ymax_xmax_format, \
    BoxFormat, apply_normalization


class PostProcessGoogleMeanAveragePrecision:
    # a converter function used to convert user boxes format to google em format (which is [ymin,xmin,ymax,xmax])
    boxes_format_converter = convert_to_ymin_xmin_ymax_xmax_format

    def __init__(self, num_categories=80, iou_list=None, boxes_format: BoxFormat = BoxFormat.XMIN_YMIN_W_H,
                 model_input_shape=(640, 640)):
        if iou_list is None:
            iou_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.num_categories = num_categories
        self.iou_list = iou_list
        self.evaluators = [ObjectDetectionEvaluation(self.num_categories, matching_iou_threshold=iou) for iou in
                           self.iou_list]
        self.results = []
        self.gt_boxes_format = boxes_format
        self.background = False
        self.batch_counter = 0

        self.H, self.W = model_input_shape
        self.min_wh, self.max_wh = 2, 7680
        self.max_nms_dets = 5000
        self.max_out_dets = 1000
        self.background = False
        self.preserve_aspect_ratio = False
        self.iou_thres = 0.65

        # Post-process parameters
        self.preserve_aspect_ratio = True
        self.iou_thres = 0.7

    def add_batch_results(self, output_annotations, target_annotations):

        image_shapes = [x['orig_img_dims'] for x in target_annotations]  # original image shape
        np_out = (np.asarray(output_annotations[0].detach().cpu()), np.asarray(output_annotations[1].detach().cpu()))
        # Post process
        post_processed_outputs = self.post_process(np_out, image_shapes,
                                                   iou_thres=self.iou_thres)

        for j in range(len(target_annotations)):
            gt_ann = target_annotations[j]
            out_ann = post_processed_outputs[j]
            if gt_ann is None:
                continue
            if len(gt_ann['boxes']) > 0:
                orig_width = target_annotations[j]['orig_img_dims'][1]
                orig_height = target_annotations[j]['orig_img_dims'][0]
                gt_boxes = PostProcessGoogleMeanAveragePrecision.boxes_format_converter(gt_ann['boxes'],
                                                                                        self.gt_boxes_format)
                gt_boxes = apply_normalization(gt_boxes, orig_width, orig_height,
                                               BoxFormat.YMIM_XMIN_YMAX_XMAX)
                [evaluator.add_single_ground_truth_image_info(gt_ann['image_id'],
                                                              gt_boxes,
                                                              np.array(gt_ann['classes']))
                 for evaluator in self.evaluators]

            if len(out_ann['boxes']) > 0:
                [evaluator.add_single_detected_image_info(gt_ann['image_id'], out_ann['boxes'],
                                                          out_ann['scores'],
                                                          np.array(out_ann['classes']))
                 for evaluator in self.evaluators]

        self.batch_counter += 1

    def post_process(self, output_annotations, image_shapes, conf_thres=0.001, iou_thres=0.65,
                     normalize_boxes=True):

        ############################################################
        # Box decoding
        ############################################################
        outputs_decoded = self.box_decoding(output_annotations)

        ############################################################
        # Post processing for each input image
        ############################################################
        # Note: outputs_decoded shape is [Batch,num_anchors*Detections,(4+1+num_categories)]
        post_processed_outputs = []
        for i, x in enumerate(outputs_decoded):
            # ----------------------------------------
            # Filter by score and width-height
            # ----------------------------------------
            scores = x[..., 4]
            wh = x[..., 2:4]
            valid_indexs = (scores > conf_thres) & ((wh > self.min_wh).any(1)) & ((wh < self.max_wh).any(1))
            x = x[valid_indexs]

            # ----------------------------------------
            # Taking Best class only
            # ----------------------------------------
            x[..., 5:] *= x[..., 4:5]  # compute confidence per class (class_score * object_score)
            conf = np.max(x[:, 5:], axis=1, keepdims=True)
            classes_id = np.argmax(x[:, 5:], axis=1, keepdims=True)

            # Change boxes format from [x_c,y_c,w,h] to [y_min,x_min,y_max,x_max]
            boxes = PostProcessGoogleMeanAveragePrecision.boxes_format_converter(x[..., :4], BoxFormat.XC_YC_W_H)
            x = np.concatenate((boxes, conf, classes_id), axis=1)[conf.reshape(-1) > conf_thres]

            # --------------------------- #
            # NMS
            # --------------------------- #
            x = x[np.argsort(-x[:, 4])[:self.max_nms_dets]]  # sort by confidence from high to low
            offset = x[..., 5:6] * np.maximum(self.H, self.W)
            boxes_offset, scores = x[..., :4] + offset, x[..., 4]  # boxes with offset by class
            valid_indexs = self.nms(boxes_offset, scores, iou_thres)
            x = x[valid_indexs]

            # --------------------------- #
            # Boxes process: scale shapes according to original input image and normalize boxes
            # --------------------------- #
            h_image, w_image = image_shapes[i][:2]
            boxes = self.scale_boxes(x[..., :4], h_image, w_image)
            boxes = apply_normalization(boxes, w_image, h_image,
                                        BoxFormat.YMIM_XMIN_YMAX_XMAX) if normalize_boxes else boxes

            # --------------------------- #
            # Classes process
            # --------------------------- #
            # convert classes from coco80 to coco91 to match labels
            classes = self.coco80_to_coco91(x[..., 5]) if self.num_categories == 80 else x[..., 5]
            classes -= 0 if self.background else 1

            # --------------------------- #
            # Scores
            # --------------------------- #
            scores = x[..., 4]

            # Add result
            post_processed_outputs.append({'boxes': boxes, 'classes': classes, 'scores': scores})

        return post_processed_outputs

    def coco80_to_coco91(self, x):  # converts 80-index to 91-index
        coco91Indexs = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
             34,
             35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
             63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90])

        return coco91Indexs[x.astype(np.int32)]

    def scale_boxes(self, boxes, h_image, w_image):
        # boxes format = [y_min, x_min, y_max, x_max]
        deltaH, deltaW = 0, 0
        scale_H, scale_W = h_image / self.H, w_image / self.W
        if self.preserve_aspect_ratio:
            scale_H = scale_W = max(h_image / self.H, w_image / self.W)
            H_tag = int(np.round(h_image / scale_H))
            W_tag = int(np.round(w_image / scale_W))
            deltaH, deltaW = int((self.H - H_tag) / 2), int((self.W - W_tag) / 2)

        # Scale and offset boxes
        boxes[..., 0] = (boxes[..., 0] - deltaH) * scale_H
        boxes[..., 1] = (boxes[..., 1] - deltaW) * scale_W
        boxes[..., 2] = (boxes[..., 2] - deltaH) * scale_H
        boxes[..., 3] = (boxes[..., 3] - deltaW) * scale_W

        # Clip boxes
        boxes = self.clip_boxes(boxes, h_image, w_image)
        return boxes

    def clip_boxes(self, boxes, h, w):
        # image_shape = [batch_size, height, width, num_channels]
        # boxes format = [y_min, x_min, y_max, x_max]
        boxes[..., 0] = np.clip(boxes[..., 0], a_min=0, a_max=h)
        boxes[..., 1] = np.clip(boxes[..., 1], a_min=0, a_max=w)
        boxes[..., 2] = np.clip(boxes[..., 2], a_min=0, a_max=h)
        boxes[..., 3] = np.clip(boxes[..., 3], a_min=0, a_max=w)
        return boxes

    def summarize_results(self):
        results = [evaluator.evaluate()[1] for evaluator in self.evaluators]
        return np.mean(np.array(results))

    def box_decoding(self, output_annotations):
        outputs_decoded = self.box_decoding_YOLO_V8(output_annotations)
        return outputs_decoded

    def box_decoding_YOLO_V8(self, prediction,
                             conf_thres=0.001,
                             nc=0,  # number of classes (optional)
                             ):

        if isinstance(prediction,
                      (list, tuple)):  # YOLOv8 model in validation model, output = (bbox, classes)
            feat_sizes = np.array([80, 40, 20])
            stride_sizes = np.array([8, 16, 32])
            anchors, strides = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))
            dbox = dist2bbox_yolo_v8(prediction[0], np.expand_dims(anchors, 0), xywh=True, dim=1) * strides
            if len(prediction) == 4:  # include masks for instance segmentation
                prediction = np.concatenate((dbox, prediction[1], prediction[2]), 1)
            else:
                prediction = np.concatenate((dbox, prediction[1]), 1)

        nc = self.num_categories  # number of classes
        mi = 4 + nc  # mask start index

        xc = np.amax(prediction[:, 4:mi], 1) > conf_thres

        z = []
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            zc = np.transpose(x)[xc[xi]]
            z.append(np.c_[zc[:, 0:4], np.ones(zc.shape[0]), zc[:, 4:]])

        return z

    def load_anchors(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            anchors = pickle.load(f)
        return anchors

    def nms(self, dets, scores, iou_thres=0.5):

        y1, x1 = dets[:, 0], dets[:, 1]
        y2, x2 = dets[:, 2], dets[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]

        return keep[:self.max_out_dets]

    def softmax(self, x):
        y = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
        z = y / np.expand_dims(np.sum(y, axis=-1), axis=-1)
        return z


def make_anchors_yolo_v8(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i], feats[i]
        sx = np.arange(stop=w) + grid_cell_offset  # shift x
        sy = np.arange(stop=h) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack((sx, sy), -1).reshape((-1, 2)))
        stride_tensor.append(np.full((h * w, 1), stride))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)


def dist2bbox_yolo_v8(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.split(distance, 2, axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), dim)  # xyxy bbox



class ObjectDetectionEvaluation(object):
    """Internal implementation of Pascal object detection metrics."""

    def __init__(self,
                 num_groundtruth_classes,
                 matching_iou_threshold=0.5,
                 nms_iou_threshold=1.0,
                 nms_max_output_boxes=10000,
                 use_weighted_mean_ap=False,
                 label_id_offset=0,
                 group_of_weight=0.0):
        if num_groundtruth_classes < 1:
            raise ValueError('Need at least 1 groundtruth class for evaluation.')

        self.per_image_eval = PerImageEvaluation(
            num_groundtruth_classes=num_groundtruth_classes,
            matching_iou_threshold=matching_iou_threshold,
            nms_iou_threshold=nms_iou_threshold,
            nms_max_output_boxes=nms_max_output_boxes,
            group_of_weight=group_of_weight)
        self.group_of_weight = group_of_weight
        self.num_class = num_groundtruth_classes
        self.use_weighted_mean_ap = use_weighted_mean_ap
        self.label_id_offset = label_id_offset

        self.groundtruth_boxes = {}
        self.groundtruth_class_labels = {}
        self.groundtruth_masks = {}
        self.groundtruth_is_difficult_list = {}
        self.groundtruth_is_group_of_list = {}
        self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=float)
        self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

        self._initialize_detections()

    def _initialize_detections(self):
        self.detection_keys = set()
        self.scores_per_class = [[] for _ in range(self.num_class)]
        self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
        self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
        self.average_precision_per_class = np.empty(self.num_class, dtype=float)
        self.average_precision_per_class.fill(np.nan)
        self.precisions_per_class = []
        self.recalls_per_class = []
        self.corloc_per_class = np.ones(self.num_class, dtype=float)

    def clear_detections(self):
        self._initialize_detections()

    def add_single_ground_truth_image_info(self,
                                           image_key,
                                           groundtruth_boxes,
                                           groundtruth_class_labels,
                                           groundtruth_is_difficult_list=None,
                                           groundtruth_is_group_of_list=None,
                                           groundtruth_masks=None):
        """Adds groundtruth for a single image to be used for evaluation.

        Args:
          image_key: A unique string/integer identifier for the image.
          groundtruth_boxes: float32 numpy array of shape [num_boxes, 4]
            containing `num_boxes` groundtruth boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
          groundtruth_class_labels: integer numpy array of shape [num_boxes]
            containing 0-indexed groundtruth classes for the boxes.
          groundtruth_is_difficult_list: A length M numpy boolean array denoting
            whether a ground truth box is a difficult instance or not. To support
            the case that no boxes are difficult, it is by default set as None.
          groundtruth_is_group_of_list: A length M numpy boolean array denoting
              whether a ground truth box is a group-of box or not. To support
              the case that no boxes are groups-of, it is by default set as None.
          groundtruth_masks: uint8 numpy array of shape
            [num_boxes, height, width] containing `num_boxes` groundtruth masks.
            The mask values range from 0 to 1.
        """
        if image_key in self.groundtruth_boxes:
            return

        self.groundtruth_boxes[image_key] = groundtruth_boxes
        self.groundtruth_class_labels[image_key] = groundtruth_class_labels
        self.groundtruth_masks[image_key] = groundtruth_masks
        if groundtruth_is_difficult_list is None:
            num_boxes = groundtruth_boxes.shape[0]
            groundtruth_is_difficult_list = np.zeros(num_boxes, dtype=bool)
        self.groundtruth_is_difficult_list[
            image_key] = groundtruth_is_difficult_list.astype(dtype=bool)
        if groundtruth_is_group_of_list is None:
            num_boxes = groundtruth_boxes.shape[0]
            groundtruth_is_group_of_list = np.zeros(num_boxes, dtype=bool)
        self.groundtruth_is_group_of_list[
            image_key] = groundtruth_is_group_of_list.astype(dtype=bool)

        self._update_ground_truth_statistics(
            groundtruth_class_labels,
            groundtruth_is_difficult_list.astype(dtype=bool),
            groundtruth_is_group_of_list.astype(dtype=bool))

    def add_single_detected_image_info(self, image_key, detected_boxes,
                                       detected_scores, detected_class_labels,
                                       detected_masks=None):
        """Adds detections for a single image to be used for evaluation.

        Args:
          image_key: A unique string/integer identifier for the image.
          detected_boxes: float32 numpy array of shape [num_boxes, 4]
            containing `num_boxes` detection boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
          detected_scores: float32 numpy array of shape [num_boxes] containing
            detection scores for the boxes.
          detected_class_labels: integer numpy array of shape [num_boxes] containing
            0-indexed detection classes for the boxes.
          detected_masks: np.uint8 numpy array of shape [num_boxes, height, width]
            containing `num_boxes` detection masks with values ranging
            between 0 and 1.

        Raises:
          ValueError: if the number of boxes, scores and class labels differ in
            length.
        """
        if (len(detected_boxes) != len(detected_scores) or
                len(detected_boxes) != len(detected_class_labels)):
            raise ValueError('detected_boxes, detected_scores and '
                             'detected_class_labels should all have same lengths. Got'
                             '[%d, %d, %d]' % len(detected_boxes),
                             len(detected_scores), len(detected_class_labels))

        if image_key in self.detection_keys:
            return

        self.detection_keys.add(image_key)
        if image_key in self.groundtruth_boxes:
            groundtruth_boxes = self.groundtruth_boxes[image_key]
            groundtruth_class_labels = self.groundtruth_class_labels[image_key]
            # Masks are popped instead of look up. The reason is that we do not want
            # to keep all masks in memory which can cause memory overflow.
            groundtruth_masks = self.groundtruth_masks.pop(
                image_key)
            groundtruth_is_difficult_list = self.groundtruth_is_difficult_list[
                image_key]
            groundtruth_is_group_of_list = self.groundtruth_is_group_of_list[
                image_key]
        else:
            groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
            groundtruth_class_labels = np.array([], dtype=int)
            if detected_masks is None:
                groundtruth_masks = None
            else:
                groundtruth_masks = np.empty(shape=[0, 1, 1], dtype=float)
            groundtruth_is_difficult_list = np.array([], dtype=bool)
            groundtruth_is_group_of_list = np.array([], dtype=bool)
        scores, tp_fp_labels, is_class_correctly_detected_in_image = (
            self.per_image_eval.compute_object_detection_metrics(
                detected_boxes=detected_boxes,
                detected_scores=detected_scores,
                detected_class_labels=detected_class_labels,
                groundtruth_boxes=groundtruth_boxes,
                groundtruth_class_labels=groundtruth_class_labels,
                groundtruth_is_difficult_list=groundtruth_is_difficult_list,
                groundtruth_is_group_of_list=groundtruth_is_group_of_list,
                detected_masks=detected_masks,
                groundtruth_masks=groundtruth_masks))

        for i in range(self.num_class):
            if scores[i].shape[0] > 0:
                self.scores_per_class[i].append(scores[i])
                self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
        self.num_images_correctly_detected_per_class += is_class_correctly_detected_in_image

    def _update_ground_truth_statistics(self, groundtruth_class_labels,
                                        groundtruth_is_difficult_list,
                                        groundtruth_is_group_of_list):
        """Update grouth truth statitistics.

        1. Difficult boxes are ignored when counting the number of ground truth
        instances as done in Pascal VOC devkit.
        2. Difficult boxes are treated as normal boxes when computing CorLoc related
        statitistics.

        Args:
          groundtruth_class_labels: An integer numpy array of length M,
              representing M class labels of object instances in ground truth
          groundtruth_is_difficult_list: A boolean numpy array of length M denoting
              whether a ground truth box is a difficult instance or not
          groundtruth_is_group_of_list: A boolean numpy array of length M denoting
              whether a ground truth box is a group-of box or not
        """
        for class_index in range(self.num_class):
            num_gt_instances = np.sum(groundtruth_class_labels[
                                          ~groundtruth_is_difficult_list
                                          & ~groundtruth_is_group_of_list] == class_index)
            num_groupof_gt_instances = self.group_of_weight * np.sum(
                groundtruth_class_labels[groundtruth_is_group_of_list] == class_index)
            self.num_gt_instances_per_class[
                class_index] += num_gt_instances + num_groupof_gt_instances
            if np.any(groundtruth_class_labels == class_index):
                self.num_gt_imgs_per_class[class_index] += 1

    def evaluate(self):
        """Compute evaluation result.

        Returns:
          A named tuple with the following fields -
            average_precision: float numpy array of average precision for
                each class.
            mean_ap: mean average precision of all classes, float scalar
            precisions: List of precisions, each precision is a float numpy
                array
            recalls: List of recalls, each recall is a float numpy array
            corloc: numpy float array
            mean_corloc: Mean CorLoc score for each class, float scalar
        """
        if self.use_weighted_mean_ap:
            all_scores = np.array([], dtype=float)
            all_tp_fp_labels = np.array([], dtype=bool)
        for class_index in range(self.num_class):
            if self.num_gt_instances_per_class[class_index] == 0:
                continue
            if not self.scores_per_class[class_index]:
                scores = np.array([], dtype=float)
                tp_fp_labels = np.array([], dtype=float)
            else:
                scores = np.concatenate(self.scores_per_class[class_index])
                tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
            if self.use_weighted_mean_ap:
                all_scores = np.append(all_scores, scores)
                all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
            precision, recall = compute_precision_recall(
                scores, tp_fp_labels, self.num_gt_instances_per_class[class_index])
            self.precisions_per_class.append(precision)
            self.recalls_per_class.append(recall)
            average_precision = compute_average_precision(precision, recall)
            self.average_precision_per_class[class_index] = average_precision

        self.corloc_per_class = compute_cor_loc(
            self.num_gt_imgs_per_class,
            self.num_images_correctly_detected_per_class)

        if self.use_weighted_mean_ap:
            num_gt_instances = np.sum(self.num_gt_instances_per_class)
            precision, recall = compute_precision_recall(
                all_scores, all_tp_fp_labels, num_gt_instances)
            mean_ap = compute_average_precision(precision, recall)
        else:
            mean_ap = np.nanmean(self.average_precision_per_class)
        mean_corloc = np.nanmean(self.corloc_per_class)
        return [
            self.average_precision_per_class, mean_ap, self.precisions_per_class,
            self.recalls_per_class, self.corloc_per_class, mean_corloc]
