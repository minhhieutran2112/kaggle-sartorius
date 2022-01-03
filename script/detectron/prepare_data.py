import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import json,itertools
import os
import random
import time
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from skimage import exposure
import torchvision.transforms as T

import scipy.ndimage as ndi
import skimage.morphology as morph
from skimage.filters import threshold_otsu

# From https://www.kaggle.com/stainsby/fast-tested-rle
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

# From https://newbedev.com/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def coco_structure(train_df,fix=False):
    cat_ids = {name:id+1 for id, name in enumerate(sorted(train_df.cell_type.unique()))}    
    cats =[{'name':name, 'id':id} for name,id in cat_ids.items()]
    images = [{'id':id, 'width':row.width, 'height':row.height, 'file_name':f'train/{id}.png'} for id,row in train_df.groupby('id').agg('first').iterrows()]
    annotations=[]
    for idx, row in tqdm(train_df.iterrows(),total=train_df.shape[0]):
        mk = rle_decode(row.annotation, (row.height, row.width))
        if fix:
            mk = mk>0
            mk, broken_mask = clean_mask(mk)
            if broken_mask:
                continue
        ys, xs = np.where(mk)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        enc =binary_mask_to_rle(mk)
        seg = {
            'segmentation':enc, 
            'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)],
            'area': int(np.sum(mk)),
            'image_id':row.id, 
            'category_id':cat_ids[row.cell_type], 
            'iscrowd':0, 
            'id':idx
        }
        annotations.append(seg)
    return {'categories':cats, 'images':images,'annotations':annotations}

TH = 40

def clean_mask(mask):
    
    mask = mask > threshold_otsu(np.array(mask).astype(np.uint8))
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)
    
    # New code for mask acceptance 
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = contours[0][:, 0]
    diff = c - np.roll(c, 1, 0)
    targets = (diff[:, 1] == 0) & (np.abs(diff[:, 0]) >= TH)  # find horizontal lines longer than threshold
    
    return mask, (True in targets)


'''each'''
for cell_type in tqdm(['shsy5y','astro','cort']):
    gkf  = GroupKFold(n_splits = 5 )
    df = pd.read_csv('/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/org_data/train.csv').query('cell_type==@cell_type')
    tmp_df=df[['id','cell_type']].drop_duplicates()
    X_train, X_test, _, _ = train_test_split(tmp_df.iloc[:,:1],tmp_df.iloc[:,1],stratify=tmp_df.iloc[:,1], test_size=0.1)
    test_df=df[df.id.isin(X_test.id)].reset_index()
    df=df[~df.id.isin(X_test.id)].reset_index(drop=True)

    test_root=coco_structure(test_df)
    with open(f'/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/cross_val_fixed/annotations_{cell_type}_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_root, f, ensure_ascii=True, indent=4)
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(gkf.split(df,groups =  np.array(df['id'].to_list())))):
        print(fold)
        train_df = df.loc[train_idx].reset_index()
        val_df = df.loc[val_idx].reset_index()
        train_root = coco_structure(train_df)
        val_root = coco_structure(val_df)
        with open(f'/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/cross_val_fixed/annotations_{cell_type}_train_fold{fold}.json', 'w', encoding='utf-8') as f:
            json.dump(train_root, f, ensure_ascii=True, indent=4)
        with open(f'/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/cross_val_fixed/annotations_{cell_type}_val_fold{fold}.json', 'w', encoding='utf-8') as f:
            json.dump(val_root, f, ensure_ascii=True, indent=4)

'''all 3 classes'''
gkf  = StratifiedGroupKFold(n_splits = 5 )
df = pd.read_csv('/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/org_data/train.csv')
tmp_df=df[['id','cell_type']].drop_duplicates()
X_train, X_test, _, _ = train_test_split(tmp_df.iloc[:,:1],tmp_df.iloc[:,1],stratify=tmp_df.iloc[:,1], test_size=0.1)
mask_dict={}
test_df=df[df.id.isin(X_test.id)].reset_index()
df=df[~df.id.isin(X_test.id)].reset_index(drop=True)

test_root=coco_structure(test_df)
with open(f'/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/cross_val_fixed/annotations_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_root, f, ensure_ascii=True, indent=4)

for fold, (train_idx, val_idx) in tqdm(enumerate(gkf.split(df,y=np.array(df['cell_type'].to_list()),groups =  np.array(df['id'].to_list())))):
    print(fold)
    train_df = df.loc[train_idx].reset_index()
    val_df = df.loc[val_idx].reset_index()
    train_root = coco_structure(train_df)
    val_root = coco_structure(val_df)
    with open(f'/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/cross_val_fixed/annotations_train_fold{fold}.json', 'w', encoding='utf-8') as f:
        json.dump(train_root, f, ensure_ascii=True, indent=4)
    with open(f'/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/cross_val_fixed/annotations_val_fold{fold}.json', 'w', encoding='utf-8') as f:
        json.dump(val_root, f, ensure_ascii=True, indent=4)

'''combine'''
combined={'images':[],'annotations':[]}
for i in tqdm(['train','test','val']) :
    if i=='test':
        img_path='/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/org_data/LIVECell_dataset_2021/images/livecell_test_images/SHSY5Y/'
    else:
        img_path='/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/org_data/LIVECell_dataset_2021/images/livecell_train_val_images/SHSY5Y/'
    with open(f'/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/org_data/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/livecell_shsy5y_{i}.json','r') as fp:
        data=json.load(fp)
    for image in data['images']:
        new_image={}
        new_image['file_name']=img_path+image['file_name']
        new_image['id']=str(image['id'])
        new_image['width']=image['width']
        new_image['height']=image['height']
        combined['images']=combined['images']+[new_image]
    for idx, annotation in data['annotations'].items():
        new_annotation={
            'segmentation':annotation['segmentation'], 
            'bbox': annotation['bbox'],
            'area': annotation['area'],
            'image_id':str(annotation['image_id']), 
            'category_id':3, 
            'iscrowd':annotation['iscrowd'], 
            'id':-annotation['id']
        }
        combined['annotations']=combined['annotations']+[new_annotation]

for fold in range(5):
    with open(f'/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/cross_val_fixed/annotations_train_fold{fold}.json', 'r') as fp:
        data=json.load(fp)
    for key in ['annotations','images']:
        data[key]=data[key]+combined[key]
    with open(f'/its/home/mt601/kaggle/sartorius-cell-instance-segmentation/cross_val_fixed/annotations_train_fold{fold}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=True, indent=4)