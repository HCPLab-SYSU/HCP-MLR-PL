import math
import numpy as np

import torch

class AverageMeter(object):
    """Compute current value, sum and average"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, num=1):
        self.val = val
        self.sum += val
        self.count += num
        self.avg = float(self.sum) / self.count if self.count != 0 else 0

class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""

        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """

        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, 'Wrong output size (should be 1D or 2D with one column per class)'

        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, 'Wrong target size (should be 1D or 2D with one column per class)'

        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), 'Dimensions for output should match previously added examples'

        # Make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            newSize = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(newSize + output.numel()))
            self.targets.storage().resize_(int(newSize + output.numel()))

        # Store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0

        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))

        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0

        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()

        # Compute average precision for each class
        for k in range(self.scores.size(1)):
            # Sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]

            # Compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)

        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # Sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        posCount, totalCount, avgPrecision = 0., 0., 0.

        for index in indices:

            if difficult_examples and target[index] == 0:
                continue

            totalCount+=1

            if target[index] == 1:
                posCount+=1
                avgPrecision+= posCount/totalCount

        return avgPrecision/posCount

    def overall(self):

        if self.scores.numel() == 0:
            return 0

        scores, targets = self.scores.cpu().numpy(), self.targets.cpu().numpy()
        targets[targets == -1] = 0

        return self.evaluation(scores, targets)

    def overall_topk(self, k):

        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0

        sampleNum, classNum = self.scores.size()
        scores = np.zeros((sampleNum, classNum)) - 1
        indexs = self.scores.topk(k, 1, True, True)[1].cpu().numpy()

        tmp = self.scores.cpu().numpy()
        for indexSample in range(sampleNum):
            for indexClass in indexs[indexSample]:
                scores[indexSample, indexClass] = 1 if tmp[indexSample, indexClass] >= 0 else -1

        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):

        sampleNum, classNum = scores_.shape
        Nc, Np, Ng = np.zeros(classNum), np.zeros(classNum), np.zeros(classNum)

        for index in range(classNum):

            scores, targets = scores_[:, index], targets_[:, index]
            targets[targets == -1] = 0

            Ng[index] = np.sum(targets == 1)
            Np[index] = np.sum(scores >= 0)
            Nc[index] = np.sum(targets * (scores >= 0))

        Np[Np == 0] = 1 

        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / classNum
        CR = np.sum(Nc / Ng) / classNum
        CF1 = (2 * CP * CR) / (CP + CR)

        return OP, OR, OF1, CP, CR, CF1

def ComputeAccuracy(output, target, topK=(1,)):
    """Compute precision@k for the specific value of k"""
   
    BatchSize, maxK = target.size()[0], max(topK)

    _, pred = output.topk(maxK, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topK:
        correctK = correct[:k].view(-1).float().sum(0)
        res.append(correctK.mut_(100.0/BatchSize))

    return res

def ComputeAP_VOC(recall, precision):
    """Compute AP with VOC standard"""

    rec, prec = np.concatenate(([0.], recall, [1.])), np.concatenate(([0.], precision, [0.]))
    for index in range(prec.size-1, 0, -1):
        prec[index-1] = np.maximum(prec[index-1], prec[index])        
    index = np.where(rec[1:]!=rec[:-1])[0]
    return np.sum((rec[index+1]-rec[index]) * prec[index+1])

def Compute_mAP_VOC2012(prediction, classNum, seenIndex=None, unseenIndex=None):
    """Compute mAP with VOC2012 standard"""

    #with open(filePath, 'r') as f:
    #    lines = f.readlines()
    #seg = np.array([line.strip().split(' ') for line in lines]).astype(float)

    Confidence, GroundTruth = prediction[:, :classNum], prediction[:, classNum:].astype(np.int32)
    APs, TP, FP = [], np.zeros(GroundTruth.shape[0]), np.zeros(GroundTruth.shape[0])

    for classId in range(classNum):

        sortedLabel = [GroundTruth[index][classId] for index in np.argsort(-Confidence[:, classId])]
        for index in range(GroundTruth.shape[0]):
            TP[index], FP[index] = (sortedLabel[index]>0), (sortedLabel[index]<=0)

        objectNum, TP, FP = sum(TP), np.cumsum(TP), np.cumsum(FP)
        recall, precision = TP / float(objectNum), TP / np.maximum(TP + FP, np.finfo(np.float64).eps)
        APs += [ComputeAP_VOC(recall, precision)]

    np.set_printoptions(precision=3, suppress=True)
    APs = np.array(APs)

    if seenIndex==None and unseenIndex==None:
        return np.mean(APs) # mAP for all
    return np.mean(APs[seenIndex]), np.mean(APs[unseenIndex]), np.mean(APs) # mAP for base, mAP for novel, mAP for all


def getConcatIndex(classNum):
    res = [[], []]
    for index in range(classNum-1):
        res[0] += [index for i in range(classNum-index-1)]
        res[1] += [i for i in range(index+1, classNum)]
    return res


def compute_pseudo_label_accuracy(meter, pseudoLabel, groundTruth, margin=0.5):

    pseudoLabel, groundTruth = pseudoLabel.cpu().data.numpy(), groundTruth.cpu().data.numpy()
    if isinstance(margin, float):
        pseudoLabel[pseudoLabel > margin] = 1
        pseudoLabel[pseudoLabel < -margin] = -1
    elif isinstance(margin, tuple):
        pseudoLabel[pseudoLabel > margin[0]] = 1
        pseudoLabel[pseudoLabel < margin[1]] = -1
    else:
        print('The margin of compute_pseudo_label_accuracy is wrong')

    meter['accuracy'].update(np.sum(pseudoLabel == groundTruth), groundTruth.shape[0] * groundTruth.shape[1])
    meter['precision'].update(np.sum((pseudoLabel == 1).astype(np.float) * (groundTruth == 1).astype(np.float)), np.sum(pseudoLabel == 1))
    meter['recall'].update(np.sum((pseudoLabel == 1).astype(np.float) * (groundTruth == 1).astype(np.float)), np.sum(groundTruth == 1))

    # Confuse Matrix
    meter['TP'].update(np.sum((pseudoLabel==1) * (groundTruth==1)), groundTruth.shape[0] * groundTruth.shape[1])
    meter['TN'].update(np.sum((pseudoLabel==-1) * (groundTruth==-1)), groundTruth.shape[0] * groundTruth.shape[1])
    meter['FP'].update(np.sum((pseudoLabel==1) * (groundTruth==-1)), groundTruth.shape[0] * groundTruth.shape[1])
    meter['FN'].update(np.sum((pseudoLabel==-1) * (groundTruth==1)), groundTruth.shape[0] * groundTruth.shape[1])


def compute_intra_cooccurrence_accuracy(meter, coOccurrence, target):
    """
    Shape of coOccurrence : (BatchSize, \sum_{i=1}^{classNum-1}{i})
    Shape of target : (BatchSize, classNum)
    """

    coOccurrence, target = torch.sigmoid(coOccurrence).cpu().data.numpy(), target.cpu().data.numpy()
    coOccurrence[coOccurrence >= 0.5] = 1
    coOccurrence[coOccurrence < 0.5] = 0

    concatIndex = getConcatIndex(target.shape[1])
    target1, target2 = target[:, concatIndex[0]], target[:, concatIndex[1]]
    target1, target2 = (target1 > 0).astype(np.float), (target2 > 0).astype(np.float)
    target = target1 * target2

    meter['accuracy'].update(np.sum(coOccurrence == target), target.shape[0] * target.shape[1])
    meter['precision'].update(np.sum((coOccurrence == 1).astype(np.float) * (target == 1).astype(np.float)), np.sum(coOccurrence == 1))
    meter['recall'].update(np.sum((coOccurrence == 1).astype(np.float) * (target == 1).astype(np.float)), np.sum(target == 1))

    totalNum = 1.0 * coOccurrence.shape[0] * coOccurrence.shape[1]

    # Confuse Matrix
    meter['TP'].update(np.sum((coOccurrence == 1) * (target == 1)), totalNum)
    meter['TN'].update(np.sum((coOccurrence == 0) * (target == 0)), totalNum)
    meter['FP'].update(np.sum((coOccurrence == 1) * (target == 0)), totalNum)
    meter['FN'].update(np.sum((coOccurrence == 0) * (target == 1)), totalNum)

    # Log FP
    meter['FP_oneExist'].update((np.sum((coOccurrence == 1) * (target1 == 1) * (target2 == 0)) + np.sum((coOccurrence == 1) * (target1 == 0) * (target2 == 1))), totalNum)
    meter['FP_noneExist'].update(np.sum((coOccurrence == 1) * (target1 == 0) * (target2 == 0)), totalNum)


def compute_inter_feature_distance(meter, feature, target):
    """
    Shape of feature : (BatchSize, classNum, featureDim)
    Shape of target : (BatchSize, classNum)
    """

    concatIndex = getConcatIndex(target.size(0))

    target_ = target.detach().clone()
    target_[target_ != 1] = 0
    pos2posTarget = target_[concatIndex[0]] * target_[concatIndex[1]]

    target_ = -1 * target.detach().clone()
    target_[target_ != 1] = 0
    neg2negTarget = target_[concatIndex[0]] * target_[concatIndex[1]]

    pos2negTarget = 1 - pos2posTarget
    pos2negTarget[(target[concatIndex[0]] == 0) | (target[concatIndex[1]] == 0)] = 0
    pos2negTarget[(target[concatIndex[0]] == -1) & (target[concatIndex[1]] == -1)] = 0

    distance = torch.nn.CosineSimilarity(dim=2, eps=1e-9)(feature[concatIndex[0]], feature[concatIndex[1]])

    if torch.sum(pos2posTarget) > 0:
        meter['pos2pos'].update(torch.mean(distance[pos2posTarget == 1]).item())
    if torch.sum(pos2negTarget) > 0:
        meter['pos2neg'].update(torch.mean(distance[pos2negTarget == 1]).item())
    if torch.sum(neg2negTarget) > 0:
        meter['neg2neg'].update(torch.mean(distance[neg2negTarget == 1]).item())