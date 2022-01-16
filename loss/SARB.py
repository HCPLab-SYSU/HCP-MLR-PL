import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(BCELoss, self).__init__()

        self.margin = margin

        self.reduce = reduce
        self.size_average = size_average

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, input, target):

        input, target = input.float(), target.float()

        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        positive_loss = self.BCEWithLogitsLoss(input, target)
        negative_loss = self.BCEWithLogitsLoss(-input, -target)

        loss = positive_mask * positive_loss + negative_mask * negative_loss

        if self.reduce:
            if self.size_average:
                return torch.mean(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.mean(loss)
            return torch.sum(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.sum(loss)
        return loss

class ContrastiveLoss(nn.Module):
    """
    Document: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, batchSize, reduce=None, size_average=None):
        super(ContrastiveLoss, self).__init__()

        self.batchSize = batchSize
        self.concatIndex = self.getConcatIndex(batchSize)

        self.reduce = reduce
        self.size_average = size_average

        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)

    def forward(self, input, target):
        """
        Shape of input: (BatchSize, classNum, featureDim)
        Shape of target: (BatchSize, classNum), Value range of target: (-1, 0, 1)
        """

        target_ = target.detach().clone()
        target_[target_ != 1] = 0
        pos2posTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        pos2negTarget = 1 - pos2posTarget
        pos2negTarget[(target[self.concatIndex[0]] == 0) | (target[self.concatIndex[1]] == 0)] = 0
        pos2negTarget[(target[self.concatIndex[0]] == -1) & (target[self.concatIndex[1]] == -1)] = 0

        target_ = -1 * target.detach().clone()
        target_[target_ != 1] = 0
        neg2negTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        distance = self.cos(input[self.concatIndex[0]], input[self.concatIndex[1]])

        if self.reduce:
            pos2pos_loss = (1 - distance)[pos2posTarget == 1]
            pos2neg_loss = (1 + distance)[pos2negTarget == 1]
            neg2neg_loss = (1 + distance)[neg2negTarget == 1]

            if pos2pos_loss.size(0) != 0:
                if neg2neg_loss.size(0) != 0:
                    neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                                              torch.sort(neg2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)
                if pos2neg_loss.size(0) != 0:
                    if pos2neg_loss.size(0) != 0:    
                        pos2neg_loss = torch.cat((torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                                                  torch.sort(pos2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)

            loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss), 0)

            if self.size_average:
                return torch.mean(loss) if loss.size(0) != 0 else torch.mean(torch.zeros_like(loss).cuda())
            return torch.sum(loss) if loss.size(0) != 0 else torch.sum(torch.zeros_like(loss).cuda())
 
        return distance

    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum - 1):
            res[0] += [index for i in range(classNum - index - 1)]
            res[1] += [i for i in range(index + 1, classNum)]
        return res