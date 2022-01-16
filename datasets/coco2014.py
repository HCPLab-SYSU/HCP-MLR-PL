import os
import sys
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..', 'cocoapi/PythonAPI'))
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets

from pycocotools.coco import COCO


class COCO2014(data.Dataset):

    def __init__(self, mode,
                 image_dir, anno_path, labels_path,
                 input_transform=None, label_proportion=1.0):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform
        self.label_proportion = label_proportion

        self.root = image_dir
        self.coco = COCO(anno_path)
        self.ids = list(self.coco.imgs.keys())
     
        with open('./data/coco/category.json','r') as load_category:
            self.category_map = json.load(load_category)

        # labels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.labels = []
        for i in range(len(self.ids)):
            img_id = self.ids[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            self.labels.append(getLabelVector(getCategoryList(target), self.category_map))
        self.labels = np.array(self.labels)
        self.labels[self.labels == 0] = -1

        # changedLabels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(-1 means label don't exist, 0 means not sure whether the label exists, 1 means label exist)
        self.changedLabels = self.labels
        if label_proportion != 1:
            print('Changing label proportion...')
            self.changedLabels = changeLabelProportion(self.labels, self.label_proportion)

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        input = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        return index, input, self.changedLabels[index], self.labels[index]

    def __len__(self):
        return len(self.ids)

# =============================================================================
# Help Functions
# =============================================================================
def getCategoryList(item):
    categories = set()
    for t in item:
        categories.add(t['category_id'])
    return list(categories)


def getLabelVector(categories, category_map):
    label = np.zeros(80)
    for c in categories:
        label[category_map[str(c)]-1] = 1.0
    return label


def getLabel(mode):

    assert mode in ('train', 'val')

    from utils.dataloader import get_data_path
    train_dir, train_anno, train_label, \
    test_dir, test_anno, test_label = get_data_path('COCO2014')

    if mode == 'train':
        image_dir, anno_path = train_dir, train_anno
    else:
        image_dir, anno_path = test_dir, test_anno

    coco = datasets.CocoDetection(root=image_dir, annFile=anno_path)
    with open('./data/coco/category.json', 'r') as load_category:
        category_map = json.load(load_category)

    labels = []
    for i in range(len(coco)):
        labels.append(getLabelVector(getCategoryList(coco[i][1]), category_map))
    labels = np.array(labels).astype(np.float64)

    np.save('./data/coco/{}_label_vectors.npy'.format(mode), labels)


def changeLabelProportion(labels, label_proportion):

    # Set Random Seed
    np.random.seed(0)

    mask = np.random.random(labels.shape)
    mask[mask < label_proportion] = 1
    mask[mask < 1] = 0
    label = mask * labels

    assert label.shape == labels.shape

    return label


def getCoOccurrenceLabel(mode):

    assert mode in ('train', 'val')

    if mode == 'train':
        label_path = './data/coco/train_label_vectors.npy'
    else:
        label_path = './data/coco/val_label_vectors.npy'

    labels = np.load(label_path).astype(np.float64)

    coOccurrenceLabel = np.zeros((labels.shape[0], sum([i for i in range(80)])), dtype=np.float64)
    for index in range(labels.shape[0]):
        correlationMatrix = labels[index][:, np.newaxis] * labels[index][np.newaxis, :]

        index_ = 0
        for i in range(80):
            for j in range(i + 1, 80):
                if correlationMatrix[i, j] > 0:
                    coOccurrenceLabel[index, index_] = 1
                index_ += 1

    np.save('./data/coco/{}_co-occurrence_label_vectors.npy'.format(mode), coOccurrenceLabel)


def getPairIndexes(labels):

    res = []
    for index in range(labels.shape[0]):
        tmp = []
        for i in range(labels.shape[1]):
            if labels[index, i] > 0:
                tmp += np.where(labels[:, i] > 0)[0].tolist()

        tmp = set(tmp)
        tmp.discard(index)
        res.append(np.array(list(tmp)))

    return res
