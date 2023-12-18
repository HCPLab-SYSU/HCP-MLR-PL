import numpy as np

import torch
import torch.nn as nn

from .backbone.resnet import resnet101
from .GraphNeuralNetwork import GatedGNN
from .SemanticDecoupling import SemanticDecoupling
from .Element_Wise_Layer import Element_Wise_Layer


class SARB(nn.Module):

    def __init__(self, adjacencyMatrix, wordFeatures, prototypeNum,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, wordFeatureDim=300, timeStep=3,
                 isAlphaLearnable=True, isBetaLearnable=True):

        super(SARB, self).__init__()

        self.backbone = resnet101()
        
        if imageFeatureDim != 2048:
            self.changeChannel = nn.Sequential(nn.Conv2d(2048, imageFeatureDim, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(imageFeatureDim),)

        self.classNum = classNum
        self.prototypeNum = prototypeNum

        self.timeStep = timeStep
        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.wordFeatureDim = wordFeatureDim
        self.imageFeatureDim = imageFeatureDim
        
        self.wordFeatures = self.load_features(wordFeatures)
        self.inMatrix, self.outMatrix = self.load_matrix(adjacencyMatrix)

        self.SemanticDecoupling = SemanticDecoupling(classNum, imageFeatureDim, wordFeatureDim, intermediaDim=intermediaDim)
        self.GraphNeuralNetwork = GatedGNN(imageFeatureDim, timeStep, self.inMatrix, self.outMatrix)

        self.fc = nn.Linear(2 * imageFeatureDim, outputDim)
        self.classifiers = Element_Wise_Layer(classNum, outputDim)

        self.cos = torch.nn.CosineSimilarity(dim=3, eps=1e-9)        
        self.prototype = []

        self.alpha = nn.Parameter(torch.tensor(0.5).float(), requires_grad=isAlphaLearnable)
        self.beta = nn.Parameter(torch.tensor(0.5).float(), requires_grad=isBetaLearnable)
 
    def forward(self, input, target=None):

        batchSize = input.size(0)

        featureMap = self.backbone(input)                                            # (batchSize, channel, imgSize, imgSize)
        if featureMap.size(1) != self.imageFeatureDim:
            featureMap = self.changeChannel(featureMap)                              # (batchSize, imgFeatureDim, imgSize, imgSize)

        semanticFeature = self.SemanticDecoupling(featureMap, self.wordFeatures)[0]  # (batchSize, classNum, outputDim)
        
        # Predict Category
        feature = self.GraphNeuralNetwork(semanticFeature) 
        output = torch.tanh(self.fc(torch.cat((feature.view(batchSize * self.classNum, -1),
                                               semanticFeature.view(-1, self.imageFeatureDim)), 1)))
        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        result = self.classifiers(output)                                            # (batchSize, classNum)

        if (not self.training):
            return result

        if target is None:
            return result, semanticFeature
            
        self.alpha.data.clamp_(min=0, max=1)
        self.beta.data.clamp_(min=0, max=1)            

        # Instance-level Mixup
        coef, mixedTarget_1 = self.mixupLabel(target, torch.flip(target, dims=[0]), self.alpha)
        coef = coef.unsqueeze(-1).repeat(1, 1, self.outputDim)
        mixedSemanticFeature_1 = coef * semanticFeature + (1-coef) * torch.flip(semanticFeature, dims=[0])

        # Predict Category
        mixedFeature_1 = self.GraphNeuralNetwork(mixedSemanticFeature_1) 
        mixedOutput_1 = torch.tanh(self.fc(torch.cat((mixedFeature_1.view(batchSize * self.classNum, -1),
                                                      mixedSemanticFeature_1.view(-1, self.imageFeatureDim)), 1)))
        mixedOutput_1 = mixedOutput_1.contiguous().view(batchSize, self.classNum, self.outputDim)
        mixedResult_1 = self.classifiers(mixedOutput_1)                          # (batchSize, classNum)

        # Prototype-level Mixup
        prototype = self.prototype[:, torch.randint(self.prototype.size(1), (1,)), :].squeeze()
        prototype = prototype.unsqueeze(0).repeat(batchSize, 1, 1)

        mask = torch.rand(target.size()).cuda()
        mask = mask * (target == 0)
        mask[torch.arange(target.size(0)), torch.argmax(mask, dim=1)] = 1
        mask[mask != 1] = 0

        mixedSemanticFeature_2 = self.beta * semanticFeature + (1-self.beta) * prototype
        mixedSemanticFeature_2 = mask.unsqueeze(-1).repeat(1, 1, self.outputDim) * mixedSemanticFeature_2 + \
                                 (1-mask).unsqueeze(-1).repeat(1, 1, self.outputDim) * semanticFeature
        mixedTarget_2 = (1-mask) * target + mask * (1-self.beta)

        # Predict Category
        mixedFeature_2 = self.GraphNeuralNetwork(mixedSemanticFeature_2)
        mixedOutput_2 = torch.tanh(self.fc(torch.cat((mixedFeature_2.view(batchSize * self.classNum, -1),
                                                      mixedSemanticFeature_2.view(-1, self.imageFeatureDim)), 1)))
        mixedOutput_2 = mixedOutput_2.contiguous().view(batchSize, self.classNum, self.outputDim)
        mixedResult_2 = self.classifiers(mixedOutput_2)                          # (batchSize, classNum)

        return result, semanticFeature, mixedResult_1, mixedTarget_1, mixedResult_2, mixedTarget_2
            
    def mixupLabel(self, label1, label2, alpha):

        matrix = torch.ones_like(label1).cuda()
        matrix[(label1 == 0) & (label2 == 1)] = alpha

        return matrix, matrix * label1 + (1-matrix) * label2

    def computePrototype(self, train_loader):

        from sklearn.cluster import KMeans

        self.eval()
        prototypes, features = [], [torch.zeros(10, self.outputDim) for i in range(self.classNum)]

        for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(train_loader):

            input, target, groundTruth = input.cuda(), target.float().cuda(), groundTruth.cuda()

            with torch.no_grad():
                featureMap = self.backbone(input)                                            # (batchSize, channel, imgSize, imgSize)
                if featureMap.size(1) != self.imageFeatureDim:
                    featureMap = self.changeChannel(featureMap)                              # (batchSize, imgFeatureDim, imgSize, imgSize)

                semanticFeature = self.SemanticDecoupling(featureMap, self.wordFeatures)[0]  # (batchSize, classNum, outputDim)

                feature = semanticFeature.cpu()

                for i in range(self.classNum):
                    features[i] = torch.cat((features[i], feature[target[:, i] == 1, i]), dim=0)

        for i in range(self.classNum):
            kmeans = KMeans(n_clusters=self.prototypeNum).fit(features[i][10:].numpy())
            prototypes.append(torch.tensor(kmeans.cluster_centers_).cuda())
        self.prototype = torch.stack(prototypes, dim=0)

    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)

    def load_matrix(self, mat):
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix, _out_matrix = nn.Parameter(torch.from_numpy(_in_matrix), requires_grad=False), nn.Parameter(torch.from_numpy(_out_matrix), requires_grad=False)
        return _in_matrix, _out_matrix

        
# =============================================================================
# Help Functions
# =============================================================================
