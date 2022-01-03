from detectron2.structures import Instances, Boxes
from detectron2.modeling import build_model
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
import pandas as pd
import logging
logging.basicConfig(filename='/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/script/detectron/shsy5y_04.log', filemode='w', level=logging.INFO)
logging.info('Start')
from scipy import ndimage
from detectron2.data.detection_utils import read_image
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from itertools import count
from ensemble_boxes import *
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.data import DatasetCatalog
import copy
import numpy as np
from contextlib import contextmanager
from itertools import count
from typing import List
import torch
from fvcore.transforms import HFlipTransform, NoOpTransform, VFlipTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.config import configurable
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    apply_augmentations,
)
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image
import copy

class DatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    @configurable
    def __init__(self, min_sizes: List[int], max_size: int, flip: bool):
        """
        Args:
            min_sizes: list of short-edge size to resize the image to
            max_size: maximum height or width of resized images
            flip: whether to apply flipping augmentation
        """
        self.min_sizes = min_sizes
        self.max_size = max_size
        self.flip = flip

    @classmethod
    def from_config(cls, cfg):
        return {
            "min_sizes": cfg.TEST.AUG.MIN_SIZES,
            "max_size": cfg.TEST.AUG.MAX_SIZE,
            "flip": cfg.TEST.AUG.FLIP,
        }


    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.

        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()

        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        # for min_size in self.min_sizes:
        #     resize = ResizeShortestEdge(min_size, self.max_size)
        #     aug_candidates.append([resize])  # resize only
        #     if self.flip:
        #         flip = RandomFlip(prob=1.0)
        #         aug_candidates.append([resize, flip])  # resize + flip
        aug_candidates.append([NoOpTransform()])
        aug_candidates.append([RandomFlip(prob=1.0,horizontal=True,vertical=False),RandomFlip(prob=1.0,horizontal=False,vertical=True)])
        aug_candidates.append([RandomFlip(prob=1.0,horizontal=False,vertical=True)])
        aug_candidates.append([RandomFlip(prob=1.0,horizontal=True,vertical=False)])

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image
            ret.append(dic)
        return ret

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfgs, THRESHOLDS=[.35, .55, .15], iou_thr=0.5, num_filters=2):
        self.models=[]
        for cfg in cfgs:
            model = build_model(cfg)
            model.eval()
            if len(cfg.DATASETS.TEST):
                self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            model.roi_heads.mask_on=True
            self.models.append(model)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        
        self.cfg=cfg.clone()
        self.input_format = cfg.INPUT.FORMAT
        self.batch_size=1
        self.THRESHOLDS=THRESHOLDS
        self.iou_thr=iou_thr
        self.tta_mapper = DatasetMapperTTA(self.cfg)
        self.num_filters=num_filters
        assert self.input_format in ["RGB", "BGR"], self.input_format
        
    @contextmanager
    def _turn_off_roi_heads(self, attrs, model_num):
        """
        Open a context where some heads in model.roi_heads are temporarily turned off.
        Args:
            attr (list[str]): the attribute in model.roi_heads which can be used
                to turn off a specific head, e.g., "mask_on", "keypoint_on".
        """
        roi_heads = self.models[model_num].roi_heads
        old = {}
        for attr in attrs:
            try:
                old[attr] = getattr(roi_heads, attr)
            except AttributeError:
                # The head may not be implemented in certain ROIHeads
                pass

        if len(old.keys()) == 0:
            yield
        else:
            for attr in old.keys():
                setattr(roi_heads, attr, False)
            yield
            for attr in old.keys():
                setattr(roi_heads, attr, old[attr])
    
    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _batch_inference(self, model, batched_inputs, detected_instances=None):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        """
        if detected_instances is None:
            detected_instances = [None] * len(batched_inputs)

        outputs = []
        inputs, instances = [], []
        for idx, input, instance in zip(count(), batched_inputs, detected_instances):
            inputs.append(input)
            instances.append(instance)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                yield model.inference(
                        inputs,
                        instances if instances[0] is not None else None,
                        do_postprocess=False,
                    )[0]
                inputs, instances = [], []
        return outputs
    
    def _get_boxes(self, model_num, model, augmented_inputs, tfms):
        orig_shape = (self.height, self.width)
        # 1: forward with all augmented images
        outputs = self._batch_inference(model, augmented_inputs)
        # 2: union the results
        all_boxes = []
        all_scores = []
        all_classes = []
#         for output in outputs:
#             # Need to inverse the transforms on boxes, to obtain results on original image
#             all_boxes.append(output.pred_boxes.tensor)
#             all_scores.extend(output.scores)
#             all_classes.extend(output.pred_classes)
#         all_boxes = torch.cat(all_boxes, dim=0)

        for output, tfm in zip(outputs, tfms):
            # Need to inverse the transforms on boxes, to obtain results on original image
            pred_boxes = output.pred_boxes.tensor
            original_pred_boxes = tfm.inverse().apply_box(pred_boxes.cpu().numpy())
            all_boxes.append(torch.from_numpy(original_pred_boxes).to(pred_boxes.device))
            all_scores.extend(output.scores)
            all_classes.extend(output.pred_classes)
        del pred_boxes, original_pred_boxes, output, outputs
        torch.cuda.empty_cache()
        # TODO: method 1: 2 filters
        if len(all_classes)==0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        all_classes=torch.stack(all_classes)
        all_scores=torch.stack(all_scores)
        if self.num_filters==2:
            pred_class=torch.mode(all_classes)[0].item()
            thresh_class=self.THRESHOLDS[pred_class]
            take=all_scores>=thresh_class
            all_scores=all_scores[take]
            all_classes=all_classes[take]
            all_boxes = torch.cat(all_boxes, dim=0)[take]
            all_classes[:]=pred_class
            del take
            torch.cuda.empty_cache()
            
            # wbf
            all_boxes[:,[0,2]]/=self.width
            all_boxes[:,[1,3]]/=self.height
            torch.cuda.empty_cache()
            
            device=all_classes.device
            all_boxes, all_scores, all_classes = weighted_boxes_fusion([all_boxes.cpu().numpy()], [all_scores.cpu().numpy()], [all_classes.cpu().numpy()], weights=[1], iou_thr=self.iou_thr, skip_box_thr=0)
            all_boxes=torch.tensor(all_boxes,device=device)
            all_scores=torch.tensor(all_scores,device=device)
            all_classes=torch.tensor(all_classes,device=device,dtype=torch.long)
            all_boxes[:,[0,2]]*=self.width
            all_boxes[:,[1,3]]*=self.height
            
#             # nms
#             merged_instances = self._merge_detections(all_boxes, all_scores, all_classes, orig_shape)
#             all_boxes=merged_instances.pred_boxes.tensor
#             all_scores=merged_instances.scores
#             all_classes=merged_instances.pred_classes
        else:
            all_boxes = torch.cat(all_boxes, dim=0)
        
        return all_boxes, all_scores, all_classes

    def _rescale_detected_boxes(self, augmented_inputs, merged_instances, tfms):
        augmented_instances = []
        for input, tfm in zip(augmented_inputs, tfms):
            # Transform the target box to the augmented image's coordinate space
            pred_boxes = merged_instances.pred_boxes.tensor.cpu().numpy()
            pred_boxes = torch.from_numpy(tfm.apply_box(pred_boxes))

            aug_instances = Instances(
                image_size=input["image"].shape[1:3],
                pred_boxes=Boxes(pred_boxes),
                pred_classes=merged_instances.pred_classes,
                scores=merged_instances.scores,
            )
            augmented_instances.append(aug_instances)
        del aug_instances, pred_boxes, merged_instances
        return augmented_instances

    def _reduce_pred_masks(self, outputs, tfms):
        # Should apply inverse transforms on masks.
        # We assume only resize & flip are used. pred_masks is a scale-invariant
        # representation, so we handle flip specially
        for output, tfm in zip(outputs, tfms):
            if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                output.pred_masks = output.pred_masks.flip(dims=[3])
            if any(isinstance(t, VFlipTransform) for t in tfm.transforms):
                output.pred_masks = output.pred_masks.flip(dims=[2])
        all_pred_masks = torch.stack([o.pred_masks for o in outputs], dim=0)
        avg_pred_masks = torch.mean(all_pred_masks, dim=0)
        return avg_pred_masks
    
    def __call__(self, fn):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            image = read_image(fn, self.input_format)
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            self.height, self.width = image.shape[:2]
            image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

            inputs = {"image": image, "height": self.height, "width": self.width}

            augmented_inputs, tfms = self._get_augmented_inputs(inputs)
            
            all_boxes, all_scores, all_classes = [], [], []
            # get box from each model
            for i, model in enumerate(self.models):
                with self._turn_off_roi_heads(["mask_on", "keypoint_on"],i):
                    # temporarily disable roi heads
                    tmp_boxes, tmp_scores, tmp_classes = self._get_boxes(i, model, augmented_inputs, tfms)
                all_boxes.append(tmp_boxes)
                all_scores.append(tmp_scores)
                all_classes.append(tmp_classes)
            del tmp_boxes, tmp_scores, tmp_classes
            torch.cuda.empty_cache()
                    
            # get most popular class
            all_boxes=torch.cat(all_boxes)
            all_classes=torch.cat(all_classes)
            all_scores=torch.cat(all_scores)
            if all_scores.shape[0]==0:
                return [], None
            pred_class=torch.mode(all_classes)[0].item()
            thresh_class=self.THRESHOLDS[pred_class]
            take=all_scores>=thresh_class
            all_scores=all_scores[take]
            all_classes=all_classes[take]
            all_classes[:]=pred_class # TODO: should we do this
            all_boxes = all_boxes[take]
            del take
            torch.cuda.empty_cache()
            # generate result
#             ## nms
#             orig_shape = (self.height, self.width)
#             merged_instances = self._merge_detections(all_boxes, all_scores, all_classes, orig_shape)
            ## wbf
            all_boxes[:,[0,2]]/=self.width
            all_boxes[:,[1,3]]/=self.height
            device=all_classes.device
            all_boxes, all_scores, all_classes = weighted_boxes_fusion([all_boxes], [all_scores], [all_classes], weights=[1], iou_thr=self.iou_thr)
            all_boxes=torch.tensor(all_boxes,device=device)
            all_scores=torch.tensor(all_scores,device=device)
            all_classes=torch.tensor(all_classes,device=device,dtype=torch.long)
            all_boxes[:,[0,2]]*=self.width
            all_boxes[:,[1,3]]*=self.height
            merged_instances = Instances((self.height, self.width))
            merged_instances.pred_boxes = Boxes(all_boxes)
            merged_instances.scores = all_scores
            merged_instances.pred_classes = all_classes
            
            del all_boxes, all_scores, all_classes
            torch.cuda.empty_cache()

            # generate augmented instance
            augmented_instances = self._rescale_detected_boxes(
                augmented_inputs, merged_instances, tfms
            )

            # generate mask
            ## run forward on the detected boxes
            i=1
            for model in self.models:
                outputs=self._batch_inference(model, augmented_inputs, augmented_instances)
                for tmp_output, tfm in zip(outputs,tfms):
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        tmp_output.pred_masks = tmp_output.pred_masks.flip(dims=[3])
                    if any(isinstance(t, VFlipTransform) for t in tfm.transforms):
                        tmp_output.pred_masks = tmp_output.pred_masks.flip(dims=[2])
                    if i==1:
                        output=tmp_output.pred_masks
                    else:
                        output+=tmp_output.pred_masks
                    i+=1
                ## average the predictions
            output/=(i-1)
            merged_instances.pred_masks = output
            merged_instances = detector_postprocess(merged_instances, self.height, self.width)
            del outputs, output,
            torch.cuda.empty_cache()
            return merged_instances, pred_class
    
    def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):
        # select from the union of all results
        num_boxes = len(all_boxes)
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # +1 because fast_rcnn_inference expects background scores as well
        all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        for idx, cls, score in zip(count(), all_classes, all_scores):
            all_scores_2d[idx, cls] = score

        merged_instances, _ = fast_rcnn_inference_single_image(
            all_boxes,
            all_scores_2d,
            shape_hw,
            1e-8,
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )
        
        del all_boxes, all_scores, all_classes
        torch.cuda.empty_cache()
        return merged_instances
    
    def _rescale_detected_boxes(self, augmented_inputs, merged_instances, tfms):
        augmented_instances = []
        for input, tfm in zip(augmented_inputs, tfms):
            # Transform the target box to the augmented image's coordinate space
            pred_boxes = merged_instances.pred_boxes.tensor.cpu().numpy()
            pred_boxes = torch.from_numpy(tfm.apply_box(pred_boxes))

            aug_instances = Instances(
                image_size=input["image"].shape[1:3],
                pred_boxes=Boxes(pred_boxes),
                pred_classes=merged_instances.pred_classes,
                scores=merged_instances.scores,
            )
            augmented_instances.append(aug_instances)
        del pred_boxes, aug_instances
        torch.cuda.empty_cache()
        return augmented_instances

import logging
import os
from collections import OrderedDict
import torch

from pathlib import Path
import random, cv2, os
import numpy as np
import pycocotools.mask as mask_util
import detectron2
import sys
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine.hooks import BestCheckpointer

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
import copy
from tqdm.auto import tqdm

dataDir=Path('/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/org_data/')
register_coco_instances('sartorius_test',{}, '/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/cross_val/annotations_test.json', dataDir)

dataset_dicts = DatasetCatalog.get('sartorius_test')

astro=[]
cort=[]
shsy5y=[]
for item in tqdm(dataset_dicts):
    if item['annotations'][0]['category_id']==0:
        astro.append(item)
    elif item['annotations'][0]['category_id']==1:
        cort.append(item)
    elif item['annotations'][0]['category_id']==2:
        shsy5y.append(item)

THRESHOLDS = [.35, .55, 0.15]

def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset, aug=False):
        super().__init__()
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset}
        self.aug=aug
        self.tp=0
        self.fp=0
        self.fn=0
        self.dataset=dataset

    def reset(self):
        self.tp=0
        self.fp=0
        self.fn=0

    def process(self, outputs):
        for inp, out in zip(self.dataset, outputs):
            if len(out) == 0:
                self.fn+=len(self.annotations_cache[inp['image_id']])
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.score(out, targ)
        return self.evaluate()

    def score(self, pred_masks, targ):
        enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
        enc_targs = list(map(lambda x:x['segmentation'], targ))
        ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, ious)
            self.tp+=tp
            self.fp+=fp
            self.fn+=fn

    def evaluate(self):
        return self.tp/(self.tp + self.fp + self.fn)

from detectron2.config import CfgNode

cfgs=[]
for fold in range(5):
    cfg = CfgNode(CfgNode.load_yaml_with_base(f"/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/model/sartorius_LIVECELL_augmented_resnest_fold{fold}/config.yaml"))
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.MODEL.WEIGHTS=f"/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/model/sartorius_LIVECELL_augmented_resnest_fold{fold}/model_best.pth"
    cfg.INPUT.MIN_SIZE_TEST=0
    cfg.TEST.DETECTIONS_PER_IMAGE=1000
    cfgs.append(cfg)

# merge mask

# def calculate_iou(ins1, ins2):
#     if not isinstance(ins1.dtype,bool):
#         ins1=ins1>=0.5
#     if not isinstance(ins2.dtype,bool):
#         ins2=ins2>=0.5
#     ovl=ins1&ins2
#     union=ins1|ins2
#     return ovl.sum()/union.sum()

# def calculate_overlap(ins1,ins2):
#     if not isinstance(ins1.dtype,bool):
#         ins1=ins1>=0.5
#     if not isinstance(ins2.dtype,bool):
#         ins2=ins2>=0.5
#     ovl=ins1&ins2
#     return ovl.sum()/ins1.sum(),ovl.sum()/ins2.sum()

# ensemble_merge_thresh_iou=0.7
# ensemble_merge_thresh_overlap=0.95

# def get_overlapped(pred_masks,threshold=0.7):
#     overlapped={}
#     for i in range(pred_masks.shape[0]):
#         for j in range(i,pred_masks.shape[0]):
#             if i==j or (i,j) in overlapped.keys():
#                 continue
#             res1,res2=calculate_overlap(pred_masks[i],pred_masks[j])
#             overlapped[(i,j)]=res1
#             overlapped[(j,i)]=res2
#     overlapped=[key for key,value in overlapped.items() if value > threshold]
#     overlapped_res=[]
#     sorted_overlapped_res=[]
#     for i in overlapped:
#         if sorted(i) not in sorted_overlapped_res:
#             overlapped_res.append(i)
#             sorted_overlapped_res.append(sorted(i))
#     return overlapped_res

# def get_iou(pred_masks,threshold=0.7):
#     iou={}
#     for i in range(pred_masks.shape[0]):
#         for j in range(i,pred_masks.shape[0]):
#             if i==j or (i,j) in iou.keys():
#                 continue
#             res=calculate_iou(pred_masks[i],pred_masks[j])
#             iou[(i,j)]=res
#     iou=[key for key,value in iou.items() if value > threshold]
#     return iou

# def merge_pred_masks_overlapped(pred_masks,inverse=True,method='max',threshold=0.7):
#     overlapped=get_overlapped(pred_masks,threshold)
#     if len(overlapped)==0:
#         return pred_masks
#     dropped=[]
#     if inverse:
#         overlapped.reverse()
#     for gr in overlapped:
#         if method=='max':
#             pred_masks[gr[0]]=np.fmax(pred_masks[gr[0]],pred_masks[gr[1]])
#         elif method=='mean':
#             ins1=pred_masks[gr[0]].copy()
#             ins2=pred_masks[gr[1]].copy()
#             ins1[ins1==0]=np.nan
#             ins2[ins2==0]=np.nan
#             pred_masks[gr[0]]=np.nan_to_num(np.nanmean(np.dstack([ins1,ins2]),axis=2))
#         dropped.append(gr[1])
#         if not inverse:
#             if gr[1] in dropped:
#                 del dropped[dropped.index(gr[1])]
#     kept=torch.tensor(list(set(range(pred_masks.shape[0])).difference(dropped)))
#     pred_masks=pred_masks[kept]
#     return merge_pred_masks_overlapped(pred_masks,threshold)

# def merge_pred_masks_iou(pred_masks,inverse=True,method='max',threshold=0.7):
#     overlapped=get_iou(pred_masks,threshold)
#     if len(overlapped)==0:
#         return pred_masks
#     dropped=[]
#     if inverse:
#         overlapped.reverse()
#     for gr in overlapped:
#         if method=='max':
#             pred_masks[gr[0]]=np.fmax(pred_masks[gr[0]],pred_masks[gr[1]])
#         elif method=='mean':
#             ins1=pred_masks[gr[0]]
#             ins2=pred_masks[gr[1]]
#             ins1[ins1==0]=np.nan
#             ins2[ins2==0]=np.nan
#             pred_masks[gr[0]]=np.nan_to_num(np.nanmean(np.dstack([ins1,ins2]),axis=2))
#         dropped.append(gr[1])
#         if not inverse:
#             if gr[1] in dropped:
#                 del dropped[dropped.index(gr[1])]
#     kept=torch.tensor(list(set(range(pred_masks.shape[0])).difference(dropped)))
#     pred_masks=pred_masks[kept]
#     return merge_pred_masks_iou(pred_masks,threshold)


def np_iou(P, T):
    if len(P.shape) == 3: P = P.reshape(P.shape[0],-1)
    if len(T.shape) == 3: T = T.reshape(T.shape[0],-1)
    P,T=P.astype(float),T.astype(float)
    inter = P@T.T
    union = P.shape[1] - (1-P)@(1-T.T)
    return inter/union

def np_overlap(P,T):
    if len(P.shape) == 3: P = P.reshape(P.shape[0],-1)
    if len(T.shape) == 3: T = T.reshape(T.shape[0],-1)
    P,T=P.astype(float),T.astype(float)
    inter = P@T.T
    return inter/P.sum(axis=1),inter/T.sum(axis=1)

def merge_pred_masks_overlapped(pred_masks,inverse=True,method='max',threshold=0.95):
    overlap,_=np_overlap(pred_masks>=0.5,pred_masks>=0.5)
    np.fill_diagonal(overlap,0)
    if overlap.max() < threshold:
        return pred_masks
    groups=list(np.where(overlap>=threshold))
    if inverse:
        groups[0],groups[1]=np.flip(groups[0]),np.flip(groups[1])
    dropped=[]
    for i,j in zip(*groups):
        if method=='max':
            pred_masks[j]=np.fmax(pred_masks[i],pred_masks[j])
        elif method=='mean':
            ins1=pred_masks[j]
            ins2=pred_masks[i]
            ins1[ins1==0]=np.nan
            ins2[ins2==0]=np.nan
            pred_masks[j]=np.nan_to_num(np.nanmean(np.dstack([ins1,ins2]),axis=2))
            np.nan_to_num(pred_masks[i],copy=False)
        dropped.append(i)
        if not inverse:
            if j in dropped:
                del dropped[dropped.index(j)]
    kept=np.array(list(set(range(pred_masks.shape[0])).difference(dropped)))
    pred_masks=pred_masks[kept]
    dropped=[]
    for i in range(len(pred_masks)):
        if (pred_masks[i]>=0.5).sum()==0:
            dropped.append(i)
    kept=np.array(list(set(range(pred_masks.shape[0])).difference(dropped)))
    pred_masks=pred_masks[kept]
    return merge_pred_masks_overlapped(pred_masks,threshold)

def merge_pred_masks_iou(pred_masks,inverse=True,method='max',threshold=0.7):
    ious=np_iou(pred_masks>=0.5,pred_masks>=0.5)
    ious=np.triu(ious,1)
    if ious.max() < threshold:
        return pred_masks
    groups=list(np.where(ious>=threshold))
    if inverse:
        groups[0],groups[1]=np.flip(groups[0]),np.flip(groups[1])
    dropped=[]
    for i,j in zip(*groups):
        if method=='max':
            pred_masks[i]=np.fmax(pred_masks[i],pred_masks[j])
        elif method=='mean':
            ins1=pred_masks[i]
            ins2=pred_masks[j]
            ins1[ins1==0]=np.nan
            ins2[ins2==0]=np.nan
            pred_masks[i]=np.nan_to_num(np.nanmean(np.dstack([ins1,ins2]),axis=2))
            np.nan_to_num(pred_masks[j],copy=False)
        dropped.append(j)
        if not inverse:
            if i in dropped:
                del dropped[dropped.index(i)]
    kept=np.array(list(set(range(pred_masks.shape[0])).difference(dropped)))
    pred_masks=pred_masks[kept]
    dropped=[]
    for i in range(len(pred_masks)):
        if (pred_masks[i]>=0.5).sum()==0:
            dropped.append(i)
    kept=np.array(list(set(range(pred_masks.shape[0])).difference(dropped)))
    pred_masks=pred_masks[kept]
    return merge_pred_masks_iou(pred_masks,threshold)

# get result
for fold in tqdm(range(1)):
    predictors=DefaultPredictor(cfgs[fold:fold+1],num_filters=1,iou_thr=0.3)
    evaluator=MAPIOUEvaluator(shsy5y) #
    config=[]
    for threshold in np.arange(0.4, 0.5, 0.1):
        predictors.THRESHOLDS[2]=threshold #
        for iou_thr in [0.3,0.5,0.7]:
            predictors.iou_thr=iou_thr
            for method in ['max']:
                for inverse in [True]:
                    preds=[]
                    for fn in [i['file_name'] for i in shsy5y]: #
                        pred, pred_class = predictors(fn)
                        preds.append(pred.pred_masks.cpu().numpy())
                    for ensemble_merge_thresh_iou in [0.5,0.75,0.9]:
                        for ensemble_merge_thresh_overlap in [0.5,0.75,0.9]:
                            for postprocess in ['after']:
                                for overlap_first in [False]:
                                    preds1=copy.deepcopy(preds)
                                    if postprocess=='before':
                                        for i in range(len(preds1)):
                                            for mask_ind in range(preds1[i].shape[0]):
                                                preds1[i][mask_ind]=ndimage.binary_fill_holes(preds1[i][mask_ind])
                                            pred_masks_corrected=np.zeros_like(preds1[i])
                                            for j in range(preds1[i].shape[0]):
                                                cont, hier = cv2.findContours(preds1[i][j].astype('uint8'),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                                drawing=np.zeros_like(preds1[i][j]).astype('uint8')
                                                for c in cont:
                                                    drawing = cv2.fillConvexPoly(drawing,points=c, color=1)
                                                pred_masks_corrected[j]=drawing>0
                                            preds1[i] = pred_masks_corrected
                                        if overlap_first:
                                            for i in range(len(preds1)):
                                                preds1[i]=merge_pred_masks_overlapped(preds1[i],inverse=inverse,method=method,threshold=ensemble_merge_thresh_overlap)
                                                preds1[i]=merge_pred_masks_iou(preds1[i],inverse=inverse,method=method,threshold=ensemble_merge_thresh_iou)>=0.5
                                        else:
                                            for i in range(len(preds1)):
                                                preds1[i]=merge_pred_masks_iou(preds1[i],inverse=inverse,method=method,threshold=ensemble_merge_thresh_iou)
                                                preds1[i]=merge_pred_masks_overlapped(preds1[i],inverse=inverse,method=method,threshold=ensemble_merge_thresh_overlap)>=0.5
                                    elif postprocess=='after':
                                        if overlap_first:
                                            for i in range(len(preds1)):
                                                preds1[i]=merge_pred_masks_overlapped(preds1[i],inverse=inverse,method=method,threshold=ensemble_merge_thresh_overlap)
                                                preds1[i]=merge_pred_masks_iou(preds1[i],inverse=inverse,method=method,threshold=ensemble_merge_thresh_iou)>=0.5
                                        else:
                                            for i in range(len(preds1)):
                                                preds1[i]=merge_pred_masks_iou(preds1[i],inverse=inverse,method=method,threshold=ensemble_merge_thresh_iou)
                                                preds1[i]=merge_pred_masks_overlapped(preds1[i],inverse=inverse,method=method,threshold=ensemble_merge_thresh_overlap)>=0.5
                                        for i in range(len(preds1)):
                                            for mask_ind in range(preds1[i].shape[0]):
                                                preds1[i][mask_ind]=ndimage.binary_fill_holes(preds1[i][mask_ind])
                                            pred_masks_corrected=np.zeros_like(preds1[i])
                                            for j in range(preds1[i].shape[0]):
                                                cont, hier = cv2.findContours(preds1[i][j].astype('uint8'),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                                drawing=np.zeros_like(preds1[i][j]).astype('uint8')
                                                for c in cont:
                                                    drawing = cv2.fillConvexPoly(drawing,points=c, color=1)
                                                pred_masks_corrected[j]=drawing>0
                                            preds1[i] = pred_masks_corrected
                                    ap=evaluator.process(preds1)
                                    config.append([threshold,1,iou_thr,method,inverse,ensemble_merge_thresh_iou,ensemble_merge_thresh_overlap,overlap_first,postprocess,ap])
                                    evaluator.reset()
                                    print([threshold,1,iou_thr,method,inverse,ensemble_merge_thresh_iou,ensemble_merge_thresh_overlap,f'overlap first {int(overlap_first)}',f'postprocess {postprocess}',ap])
                                    logging.info(f'{threshold},1,{iou_thr},{method},{inverse},{ensemble_merge_thresh_iou},{ensemble_merge_thresh_overlap},overlap first {int(overlap_first)},postprocess {postprocess},{ap}')
    df=pd.DataFrame(config,columns=['threshold','num_filter','iou_thr','method','inverse','ensemble_merge_thresh_iou','ensemble_merge_thresh_overlap','overlap first','postprocess','ap'])
    print(df)
    df.to_csv(f'/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/script/detectron/shsy5y_04_configs_fold{fold}.csv',index=False)