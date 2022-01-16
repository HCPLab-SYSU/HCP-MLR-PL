import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticDecoupling(nn.Module):

    def __init__(self, classNum, imgFeatureDim, wordFeatureDim, intermediaDim=1024):
        
        super(SemanticDecoupling, self).__init__()

        self.classNum = classNum
        self.imgFeatureDim = imgFeatureDim
        self.wordFeatureDim = wordFeatureDim
        self.intermediaDim = intermediaDim

        self.fc1 = nn.Linear(self.imgFeatureDim, self.intermediaDim, bias=False)
        self.fc2 = nn.Linear(self.wordFeatureDim, self.intermediaDim, bias=False)
        self.fc3 = nn.Linear(self.intermediaDim, self.intermediaDim)
        self.fc4 = nn.Linear(self.intermediaDim, 1)

    def forward(self, imgFeaturemap, wordFeatures, visualize=False):
        '''
        Shape of imgFeaturemap : (BatchSize, Channel, imgSize, imgSize)
        Shape of wordFeatures : (classNum, wordFeatureDim)
        '''

        BatchSize, imgSize = imgFeaturemap.size()[0], imgFeaturemap.size()[3]
        imgFeaturemap = torch.transpose(torch.transpose(imgFeaturemap, 1, 2), 2, 3) # BatchSize * imgSize * imgSize * Channel
        
        imgFeature = imgFeaturemap.contiguous().view(BatchSize * imgSize * imgSize, -1)                                             # (BatchSize * imgSize * imgSize) * Channel
        imgFeature = self.fc1(imgFeature).view(BatchSize * imgSize * imgSize, 1, -1).repeat(1, self.classNum, 1)                    # (BatchSize * imgSize * imgSize) * classNum * intermediaDim
        wordFeature = self.fc2(wordFeatures).view(1, self.classNum, self.intermediaDim).repeat(BatchSize * imgSize * imgSize, 1, 1) # (BatchSize * imgSize * imgSize) * classNum * intermediaDim
        feature = self.fc3(torch.tanh(imgFeature * wordFeature).view(-1, self.intermediaDim))                                       # (BatchSize * imgSize * imgSize * classNum) * intermediaDim
        
        Coefficient = self.fc4(feature)                                                                                             # (BatchSize * imgSize * imgSize * classNum) * 1
        Coefficient = torch.transpose(torch.transpose(Coefficient.view(BatchSize, imgSize, imgSize, self.classNum), 2, 3), 1, 2).view(BatchSize, self.classNum, -1)
        Coefficient = F.softmax(Coefficient, dim=2)                                                                                 # BatchSize * classNum * (imgSize * imgSize))
        Coefficient = Coefficient.view(BatchSize, self.classNum, imgSize, imgSize)                                                  # BatchSize * classNum * imgSize * imgSize
        Coefficient = torch.transpose(torch.transpose(Coefficient, 1, 2), 2, 3)                                                     # BatchSize * imgSize * imgSize * classNum
        Coefficient = Coefficient.view(BatchSize, imgSize, imgSize, self.classNum, 1).repeat(1, 1, 1, 1, self.imgFeatureDim)        # BatchSize * imgSize * imgSize * classNum * imgFeatureDim

        featuremapWithCoefficient = imgFeaturemap.view(BatchSize, imgSize, imgSize, 1, self.imgFeatureDim).repeat(1, 1, 1, self.classNum, 1) * Coefficient # BatchSize * imgSize * imgSize * classNum * imgFeatureDim
        semanticFeature = torch.sum(torch.sum(featuremapWithCoefficient, 1), 1)                                                                            # BatchSize * classNum * imgFeatureDim

        if visualize:
            return semanticFeature, torch.sum(torch.abs(featuremapWithCoefficient), 4), Coefficient[:,:,:,:,0]
        return semanticFeature


