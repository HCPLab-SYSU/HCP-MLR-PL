import os
import sys
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets


class VG(data.Dataset):

    def __init__(self, mode,
                 image_dir, anno_path, labels_path,
                 input_transform=None, label_proportion=1.0):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform
        self.label_proportion = label_proportion

        self.img_dir = image_dir
        self.imgName_path = anno_path
        self.img_names = open(self.imgName_path, 'r').readlines()

        # labels : numpy.ndarray, shape->(len(vg), 200)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.labels_path = labels_path
        _ = json.load(open(self.labels_path, 'r'))
        self.labels = np.zeros((len(self.img_names), 200)).astype(np.int) - 1
        for i in range(len(self.img_names)):
            self.labels[i][_[self.img_names[i][:-1]]] = 1

        # changedLabels : numpy.ndarray, shape->(len(vg), 200)
        # value range->(-1 means label don't exist, 0 means not sure whether the label exists, 1 means label exist)
        self.changedLabels = self.labels
        if label_proportion != 1:
            print('Changing label proportion...')
            self.changedLabels = changeLabelProportion(self.labels, self.label_proportion)

    def __getitem__(self, index):
        name = self.img_names[index][:-1]
        input = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        if self.input_transform:
           input = self.input_transform(input)
        return index, input, self.changedLabels[index], self.labels[index]

    def __len__(self):
        return len(self.img_names)

# =============================================================================
# Help Functions
# =============================================================================
def changeLabelProportion(labels, label_proportion):

    # Set Random Seed
    np.random.seed(0)

    mask = np.random.random(labels.shape)
    mask[mask < label_proportion] = 1
    mask[mask < 1] = 0
    label = mask * labels

    assert label.shape == labels.shape

    return label


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
