# %%
import sys
sys.path.remove('/its/home/mt601/.local/lib/python3.7/site-packages')

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
wandb.login(key='6fa032c9de89fb7104cc7828743a6ad1ece62906')

import os
import time
import random
import collections
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import cv2
import albumentations as A
from albumentations import Normalize as NormalizeA
from albumentations import Resize as ResizeA
from albumentations import Compose as ComposeA
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
from types import MethodType

import torch
from torch import nn
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchmetrics

pl.seed_everything(2021)

# %%
PARENT_DIR='/mnt/maths/mt601/kaggle/sartorius-cell-instance-segmentation/'
DATA_DIR=PARENT_DIR+'data/'
TRAIN_PATH=DATA_DIR+'train/'
TEST_PATH=DATA_DIR+'test/'
MODEL_PATH=PARENT_DIR+'model/'
CLASSIFIER_PATH=MODEL_PATH+'classifier/'

CELL_TYPES  = {0: 'shsy5y', 1: 'astro', 2: 'cort'}
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)
IMAGE_RESIZE = (224, 224)

# The maximum possible amount of predictions
# 539 is the 90% percentile of the cell_type with more instances per image
BOX_DETECTIONS_PER_IMG = 559

# %%
df_base=pd.read_csv(DATA_DIR+'train.csv')
sample_sub=pd.read_csv(DATA_DIR+'sample_submission.csv')

# %% [markdown]
# # Configuration

# %%
# Reduced the train dataset to 5000 rows
TEST = False

BATCH_SIZE = 2
NUM_EPOCHS = 30

WIDTH = 704
HEIGHT = 520

resize_factor = False # 0.5

# Normalize to resnet mean and std if True.
NORMALIZE = False
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

# No changes tried with the optimizer yet.
MOMENTUM = 0.9
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

# Changes the confidence required for a pixel to be kept for a mask. 
# Only used 0.5 till now.
# MASK_THRESHOLD = 0.5
# MIN_SCORE = 0.5
# cell type specific thresholds
cell_type_dict = {"astro": 1, "cort": 2, "shsy5y": 3}
# mask_threshold_dict = {1: 0.55, 2: 0.75, 3: 0.45}
# min_score_dict = {1: 0.55, 2: 0.75, 3: 0.6}
mask_threshold_dict = {1: 0.55, 2: 0.75, 3: 0.45}
min_score_dict = {1: 0.35, 2: 0.55, 3: 0.15}
min_area={1:150,2:75,3:75}

# Use a StepLR scheduler if True. 
USE_SCHEDULER = False

PCT_IMAGES_VALIDATION = 0.075

BOX_DETECTIONS_PER_IMG = 540

# %% [markdown]
# # Utilities

# %%
# ref: https://www.kaggle.com/inversion/run-length-decoding-quick-start
def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height, width, channels) of array to return
    color: color for the mask
    Returns numpy array (mask)

    '''
    s = mask_rle.split()

    starts = list(map(lambda x: int(x) - 1, s[0::2]))
    lengths = list(map(int, s[1::2]))
    ends = [x + y for x, y in zip(starts, lengths)]
    if len(shape)==3:
        img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    else:
        img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for start, end in zip(starts, ends):
        img[start : end] = color

    return img.reshape(shape)


def rle_encoding(x):
    dots = np.where(x.cpu().numpy().flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))

# TODO: remove pixels of mask with less confidence
def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if torch.sum(mask & other_mask) > 0:
            mask[mask & other_mask] = 0
    return mask

def combine_masks(masks, mask_threshold):
    """
    combine masks into one image
    """
    device=model.device
    maskimg = torch.zeros((HEIGHT, WIDTH),device=device)
    # print(len(masks.shape), masks.shape)
    for m, mask in enumerate(masks,1):
        maskimg[mask>mask_threshold] = m
    return maskimg

# TODO: remove pixels of mask with less confidence
# TODO: change threshold for low-scoring results
def get_filtered_masks(pred,label):
    """
    filter masks using MIN_SCORE for mask and MAX_THRESHOLD for pixels
    """
    use_masks = []   
    for i, mask in enumerate(pred["masks"]):

        # Filter-out low-scoring results. Not tried yet.
        scr = pred["scores"][i].cpu().item()
        if scr > min_score_dict[label]:
            mask = mask.squeeze()
            # Keep only highly likely pixels
            binary_mask = mask > mask_threshold_dict[label]
            binary_mask = remove_overlapping_pixels(binary_mask, use_masks) # TODO: remove pixels of mask with less confidence
            use_masks.append(binary_mask)

    # TODO: find all mask then remove later
    return use_masks

# %%
def compute_iou(labels, y_pred, verbose=0):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        labels (np array): Labels.
        y_pred (np array): predictions

    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """

    if isinstance(labels,torch.Tensor):
        labels=labels.cpu().numpy()
    
    if isinstance(y_pred,torch.Tensor):
        y_pred=y_pred.cpu().numpy()

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    if verbose:
        print("Number of true objects: {}".format(true_objects))
        print("Number of predicted objects: {}".format(pred_objects))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
    intersection = intersection[1:, 1:] # exclude background
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    iou = intersection / union
    
    return iou  

def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn

def iou_map(truths, preds, verbose=0):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        truths (list of masks): Ground truths.
        preds (list of masks): Predictions.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """
    ious = [compute_iou(truth, pred, verbose) for truth, pred in zip(truths, preds)]

    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)


def get_score(ds, mdl):
    """
    Get average IOU mAP score for a dataset
    """
    mdl.eval()
    iouscore = 0
    for i in tqdm(range(len(ds))):
        img, targets = ds[i]
        with torch.no_grad():
            result = mdl([img.to(DEVICE)])[0]
            
        masks = combine_masks(targets['masks'], 0.5)
        labels = pd.Series(result['labels'].cpu().numpy()).value_counts()

        mask_threshold = mask_threshold_dict[labels.sort_values().index[-1]]
        pred_masks = combine_masks(get_filtered_masks(result), mask_threshold)
        iouscore += iou_map([masks],[pred_masks])
    return iouscore / len(ds)

# %%
def get_score(detection,target,label):
    masks = combine_masks(target['masks'], 0.5)
    mask_threshold=mask_threshold_dict[label]
    pred_masks = combine_masks(get_filtered_masks(detection,label), mask_threshold)
    return iou_map([masks],[pred_masks])

class CompetitionScore(torchmetrics.Metric):
    """
    Directly optimizes the competition metric
    """
    def __init__(self):
        super().__init__()
        self.add_state("iouscore", torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state("n_records", torch.tensor(0), dist_reduce_fx='sum')
    def update(self, detections, targets,labels):
        groups=zip(detections,targets,labels)
        for detection,target,label in groups:
            self.iouscore += get_score(detection,target,label)
        self.n_records += len(targets)
    def compute(self):
        return self.iouscore / self.n_records

# %% [markdown]
# ## Transformation

# %% [markdown]
# Just Horizontal and Vertical Flip for now.
# 
# Normalization to Resnet's mean and std can be performed using the parameter NORMALIZE in the top cell.
# 
# The first 3 transformations come from this utils package by Abishek, VerticalFlip is my adaption of HorizontalFlip, and Normalize is of my own.

# %%
# These are slight redefinitions of torch.transformation classes
# The difference is that they handle the target and the mask
# Copied from Abishek, added new ones
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class VerticalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-2)
        return image, target

class HorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        return image, target

class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, RESNET_MEAN, RESNET_STD)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    

def get_transform(train):
    transforms = [ToTensor()]
    if NORMALIZE:
        transforms.append(Normalize())
    
    # Data augmentation for train
    if train: 
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))

    return Compose(transforms)

# %% [markdown]
# # Training Dataset and DataLoader

# %%
classifier=torch.load(CLASSIFIER_PATH+'resnet34-finetuned.bin')
for param in classifier.parameters():
    param.requires_grad = False
classifier.eval()
classifier.to(DEVICE)

class CellDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None, resize=False):
        self.transforms = transforms
        self.classifier_transforms=ComposeA([ResizeA(IMAGE_RESIZE[0], IMAGE_RESIZE[1]), NormalizeA(mean=RESNET_MEAN, std=RESNET_STD, p=1), ToTensorV2()])
        self.image_dir = image_dir
        self.df = df
        
        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(HEIGHT * resize)
            self.width = int(WIDTH * resize)
            print("image size used:", self.height, self.width)
        else:
            self.height = HEIGHT
            self.width = WIDTH
        
        self.image_info = collections.defaultdict(dict)
        temp_df = self.df.groupby(["id", "cell_type"])['annotation'].agg(lambda x: list(x)).reset_index()
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                    'image_id': row['id'],
                    'image_path': os.path.join(self.image_dir, row['id'] + '.png'),
                    'annotations': list(row["annotation"]),
                    'cell_type': cell_type_dict[row["cell_type"]]
                    }
            
    def get_box(self, a_mask):
        ''' Get the bounding box of a given mask '''
        pos = np.where(a_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        ''' Get the image and the target'''
        
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        classifier_img = self.classifier_transforms(image=cv2.imread(img_path))['image']
        
        if self.should_resize:
            img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[idx]

        n_objects = len(info['annotations'])
        masks = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        boxes = []
        labels = []
        for i, annotation in enumerate(info['annotations']):
            a_mask = rle_decode(annotation, (HEIGHT, WIDTH))
            a_mask = Image.fromarray(a_mask)
            
            if self.should_resize:
                a_mask = a_mask.resize((self.width, self.height), resample=Image.BILINEAR)
            
            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask
            
            boxes.append(self.get_box(a_mask))

        # labels
        labels = [1 for _ in range(n_objects)]
        #labels = [1 for _ in range(n_objects)]
        
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        # This is the required target for the Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return {
            'label':classifier(classifier_img.unsqueeze(0).to(DEVICE)).argmax(dim=1)[0].cpu().item(),
            'img':img, 
            'target':target
        }

    def __len__(self):
        return len(self.image_info)

def collate_fn(batch):
    img_cort=[]
    target_cort=[]
    img_shsy5y=[]
    target_shsy5y=[]
    img_astro=[]
    target_astro=[]
    for item in batch:
        if item['label']==0:
            img_shsy5y.append(item['img'])
            target_shsy5y.append(item['target'])
        elif item['label']==1:
            img_astro.append(item['img'])
            target_astro.append(item['target'])
        else:
            img_cort.append(item['img'])
            target_cort.append(item['target'])
    return {
        'img_shsy5y':img_shsy5y,
        'target_shsy5y':target_shsy5y,
        'img_astro':img_astro,
        'target_astro':target_astro,
        'img_cort':img_cort,
        'target_cort':target_cort,
    }

# %%
df_images = df_base.groupby(["id", "cell_type"]).agg({'annotation': 'count'}).sort_values("annotation", ascending=False).reset_index()

for ct in cell_type_dict:
    ctdf = df_images[df_images["cell_type"]==ct].copy()
    if len(ctdf)>0:
        ctdf['quantiles'] = pd.qcut(ctdf['annotation'], 5)

# %%
df_images.groupby("cell_type").annotation.describe().astype(int)

# %%
# We used this as a reference to fill BOX_DETECTIONS_PER_IMG=140
df_images[['annotation']].describe().astype(int)

# %%
# Use the quantiles of amoount of annotations to stratify
df_images_train, df_images_test = train_test_split(df_images, stratify=df_images['cell_type'], test_size=PCT_IMAGES_VALIDATION)
df_images_train, df_images_val = train_test_split(df_images_train, stratify=df_images_train['cell_type'], test_size=PCT_IMAGES_VALIDATION)
df_train = df_base[df_base['id'].isin(df_images_train['id'])]
df_val = df_base[df_base['id'].isin(df_images_val['id'])]
df_test = df_base[df_base['id'].isin(df_images_test['id'])]
print(f"Images in train set:           {len(df_images_train)}")
print(f"Annotations in train set:      {len(df_train)}")
print(f"Images in validation set:      {len(df_images_val)}")
print(f"Annotations in validation set: {len(df_val)}")
print(f"Images in test set:            {len(df_images_test)}")
print(f"Annotations in test set:       {len(df_test)}")

# %% [markdown]
# # Train Model

# %%
def get_model(num_classes, BOX_DETECTIONS_PER_IMG, model_chkpt=None):
    # This is just a dummy value for the classification head
    
    if NORMALIZE:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                   box_detections_per_img=BOX_DETECTIONS_PER_IMG,
                                                                   image_mean=RESNET_MEAN,
                                                                   image_std=RESNET_STD)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                   box_detections_per_img=BOX_DETECTIONS_PER_IMG)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes+1)
    
    if model_chkpt:
        model.load_state_dict(torch.load(model_chkpt, map_location=DEVICE))
    return model

def model_forward(self, images, targets=None):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).

    """
    if len(images)!=0:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                            "of shape [N, 4], got {:}.".format(
                                                boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                        "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                        " Found invalid box {} for target at index {}."
                                        .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
    
    else:
        detections=[]
        losses={}
    return losses, detections


# Get the Mask R-CNN model
# The model does classification, bounding boxes and MASKs for individuals, all at the same time
# We only care about MASKS

# Current: choose mask threshold by using the threshold for cell_type with highest number of blobs
# TODO: Change to add a classification model | a separate model for each cell type

class CustomModel(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf)
        self.ds_train = CellDataset(TRAIN_PATH, df_train, resize=resize_factor, transforms=get_transform(train=True))
        self.ds_val = CellDataset(TRAIN_PATH, df_val, resize=resize_factor, transforms=get_transform(train=False))

        
        self.cort_model=get_model(self.hparams.num_classes, self.hparams.num_box_cort, self.hparams.model_chkpt)
        self.cort_model.forward=MethodType(model_forward,self.cort_model)
        for param in self.cort_model.parameters():
            param.requires_grad = True

        self.shsy5y_model=get_model(self.hparams.num_classes, self.hparams.num_box_shsy5y, self.hparams.model_chkpt)
        self.shsy5y_model.forward=MethodType(model_forward,self.shsy5y_model)
        for param in self.shsy5y_model.parameters():
            param.requires_grad = True
        
        self.astro_model=get_model(self.hparams.num_classes, self.hparams.num_box_astro, self.hparams.model_chkpt)
        self.astro_model.forward=MethodType(model_forward,self.astro_model)
        for param in self.astro_model.parameters():
            param.requires_grad = True

        self.val_metric=CompetitionScore()

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    def forward(self,batch):
        output_cort = self.cort_model(batch['img_cort'],batch['target_cort'])
        output_astro = self.astro_model(batch['img_astro'],batch['target_astro'])
        output_shsy5y = self.shsy5y_model(batch['img_shsy5y'],batch['target_shsy5y'])
        detections=output_cort[1]+output_astro[1]+output_shsy5y[1]
        losses={}
        for i in set(list(output_cort[0].keys())+list(output_astro[0].keys())+list(output_shsy5y[0].keys())):
            losses[i]=output_cort[0].get(i,0)+output_astro[0].get(i,0)+output_shsy5y[0].get(i,0)
        return losses,detections

    def configure_optimizers(self):
        params = [p for p in self.cort_model.parameters() if p.requires_grad] + [p for p in self.shsy5y_model.parameters() if p.requires_grad] + [p for p in self.astro_model.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer__name(params, lr=self.hparams.optimizer__lr, weight_decay=self.hparams.optimizer__weight_decay, momentum=self.hparams.optimizer__momentum)
        lr_scheduler = self.hparams.lr_scheduler__name(optimizer, step_size=self.hparams.lr_scheduler__step_size, gamma=self.hparams.lr_scheduler__gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler" : lr_scheduler
        } if self.hparams.use_scheduler else optimizer

    def training_step(self,batch,batch_idx):
        loss_dict, _ = self.forward(batch)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss',loss,on_epoch=True,on_step=True)
        self.log('train_loss_classifier',loss_dict['loss_classifier'],on_epoch=True,on_step=True)
        self.log('train_loss_mask',loss_dict['loss_mask'],on_epoch=True,on_step=True)
        return loss

    def validation_step(self,batch,batch_idx):
        _, detections = self.forward(batch)
        labels=[cell_type_dict['cort']]*len(batch['target_cort'])+[cell_type_dict['astro']]*len(batch['target_astro'])+[cell_type_dict['shsy5y']]*len(batch['target_shsy5y'])
        self.val_metric(detections,batch['target_cort']+batch['target_astro']+batch['target_shsy5y'],labels)
    
    def validation_epoch_end(self, outs):
        self.log('val_comp_metric', self.val_metric.compute())

class WandbImageCallback(pl.Callback):
    """Logs the input images and output predictions of a module.
    
    Predictions and labels are logged as class indices."""
    
    def __init__(self, val_ds, num_samples=10):
        super().__init__()
        self.ds=Subset(val_ds, np.arange(num_samples))

    def create_image(self,img):
        label_hue = np.uint8(179*img/img.max())
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue==0] = 0
        labeled_img=labeled_img.astype(np.float32)
        return labeled_img/255

    def pad_img(self,img):
        new_img=np.ones((img.shape[0],int(np.ceil(img.shape[1]*1.1)),img.shape[2]),dtype=np.float32)
        new_img[:,:img.shape[1],:]=img
        return new_img

    def on_validation_end(self, trainer, pl_module):
        captions=[]
        for i,img in enumerate(self.ds):
            image = img['img'].to(pl_module.device)
            target = {k: v.to(pl_module.device) for k, v in img['target'].items()}
            
            if img['label']==0:
                result = pl_module.shsy5y_model([image])[1][0]
                label=cell_type_dict['shsy5y']
            elif img['label']==1:
                result = pl_module.astro_model([image])[1][0]
                label=cell_type_dict['astro']
            else:
                result = pl_module.cort_model([image])[1][0]
                label=cell_type_dict['cort']

            masks = combine_masks(target['masks'], 0.5)
            pred_masks = combine_masks(get_filtered_masks(result,label), mask_threshold_dict[label])

            captions.append(iou_map([masks],[pred_masks]))

            masks=torch.tensor(self.pad_img(self.create_image(masks.cpu()).transpose((2,0,1))),dtype=image.dtype)
            pred_masks=torch.tensor(self.create_image(pred_masks.cpu()).transpose((2,0,1)),dtype=image.dtype)
            image=torch.tensor(self.pad_img(image.cpu()))

            if i==0:
                mosaics=torch.cat([image,masks,pred_masks],axis=1).unsqueeze(0)
            else:
                mosaics=torch.cat([mosaics,torch.cat([image,masks,pred_masks],axis=1).unsqueeze(0)],axis=0)
            
        trainer.logger.experiment.log({
            "val/examples": [wandb.Image(mosaic, caption='Score: {:.5f}'.format(caption)) 
                              for mosaic,caption in zip(mosaics,captions)],
            "global_step": trainer.global_step
            })

val_ds=CellDataset(TRAIN_PATH, df_val, resize=resize_factor, transforms=get_transform(train=False))

# %%
CONFIG={
    'num_classes':1, 
    'batch_size':4, 
    'model_chkpt':None,
    'optimizer__name':torch.optim.SGD,
    'optimizer__lr':0.001,
    'optimizer__momentum':0.9,
    'optimizer__weight_decay':0.0005,
    'use_scheduler': False,
    'lr_scheduler__name':torch.optim.lr_scheduler.StepLR,
    'lr_scheduler__step_size':0.5,
    'lr_scheduler__gamma':0.1,
    'max_epochs':30,
    'num_box_cort':110,
    'num_box_astro':600,
    'num_box_shsy5y':800,
}

# %%
checkpoint_callback = ModelCheckpoint(monitor="val_comp_metric",mode='max')
early_stop_callback = EarlyStopping(monitor='val_comp_metric',mode='max',patience=5)
image_callback = WandbImageCallback(val_ds)

group_name='Supervised'
job_type='separate_model'
tags='baseline'
name='baseline'

model=CustomModel(CONFIG)
wandb_logger = WandbLogger(project='sartorius',config=CONFIG,group=group_name,job_type=job_type,name=name,log_model=True,tags=tags)
wandb_logger.watch(model,'all',5)
trainer = pl.Trainer(
    # auto_scale_batch_size=True,
    # accelerator='ddp',
    gpus=1,
    precision=16,
    max_epochs=CONFIG['max_epochs'],
    auto_select_gpus=True,
    callbacks=[
        checkpoint_callback,
        # early_stop_callback,
        image_callback
    ],
    logger=wandb_logger,
    log_every_n_steps=5)
# trainer.tune(model)
trainer.fit(model)
