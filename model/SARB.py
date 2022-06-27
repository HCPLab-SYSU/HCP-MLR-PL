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


class SARB_journal(nn.Module):

    def __init__(self, adjacencyMatrix, wordFeatures, topK,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, wordFeatureDim=300, timeStep=3,
                 alpha=0.5, isAlphaLearnable=True, 
                 beta=0.5, isBetaLearnable=True):

        super(SARB_journal, self).__init__()

        self.backbone = resnet101()

        if imageFeatureDim != 2048:
            self.changeChannel = nn.Sequential(nn.Conv2d(2048, imageFeatureDim, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(imageFeatureDim),)

        self.classNum = classNum
        self.topK = topK

        self.poolingType = poolingType
        self.avgPooling = nn.AdaptiveAvgPool2d(1)
        self.maxPooling = nn.AdaptiveMaxPool2d(1)

        self.timeStep = timeStep
        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.wordFeatureDim = wordFeatureDim
        self.imageFeatureDim = imageFeatureDim
        
        self.wordFeatures = self.load_features(wordFeaturesPath)
        self.inMatrix, self.outMatrix = self.load_matrix(adjacencyMatrixPath)

        self.SemanticDecoupling = SemanticDecoupling(classNum, imageFeatureDim, wordFeatureDim, intermediaDim=intermediaDim, poolingType=poolingType) 
        self.GraphNeuralNetwork = GatedGNN(imageFeatureDim, timeStep, self.inMatrix, self.outMatrix) 

        self.fc = nn.Linear(2 * imageFeatureDim, outputDim)
        self.classifiers = Element_Wise_Layer(classNum, outputDim)

        self.cos = torch.nn.CosineSimilarity(dim=3, eps=1e-9)        

        # propotype_feature: (classNum, prototypeNum, imageFeatureDim)
        # prototype_featuremap: (classNum, 4, imgSize, imgSize, imageFeatureDim)

        self.alpha = nn.Parameter(torch.tensor(0.5).float(), requires_grad=isAlphaLearnable) 
        self.beta = nn.Parameter(torch.tensor(0.5).float(), requires_grad=isBetaLearnable)


    def forward(self, input, target=None):

        batchSize = input.size(0)

        featureMap = self.backbone(input)                                            # (batchSize, channel, imgSize, imgSize)
        if featureMap.size(1) != self.imageFeatureDim:
            featureMap = self.changeChannel(featureMap)                              # (batchSize, imgFeatureDim, imgSize, imgSize)

        semanticFeature, featuremapWithCoef, coefficient = self.SemanticDecoupling(featureMap, self.wordFeatures)     # (batchSize, classNum, outputDim)
        
        # Predict Category
        feature = self.GraphNeuralNetwork(semanticFeature) if self.useGatedGNN else semanticFeature
        output = torch.tanh(self.fc(torch.cat((feature.view(batchSize * self.classNum, -1),
                                               semanticFeature.view(-1, self.imageFeatureDim)),1)))

        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        result = self.classifiers(output)                                            # (batchSize, classNum)

        if onlyFeature:
            return semanticFeature, featuremapWithCoef, coefficient

        if (not self.training):
            return result

        if target is None:
            return result, semanticFeature
        
        self.alpha.data.clamp_(min=0.0, max=1.0)
        self.beta.data.clamp_(min=0.0, max=1.0) 

        # instance level
        mixedSemanticFeature_1, mixedTarget_1 = self.mixupInstance_featuremap(featuremapWithCoef, target)
        mixedFeature_1 = self.GraphNeuralNetwork(mixedSemanticFeature_1)
        mixedOutput_1 = torch.tanh(self.fc(torch.cat((mixedFeature_1.view(batchSize * self.classNum, -1),
                                                      mixedSemanticFeature_1.view(-1, self.imageFeatureDim)),1)))
        mixedOutput_1 = mixedOutput_1.contiguous().view(batchSize, self.classNum, self.outputDim)
        mixedResult_1 = self.classifiers(mixedOutput_1)                          # (batchSize, classNum)

        # prototype level
        mixedSemanticFeature_2, mixedTarget_2 = self.mixupPrototype_featuremap(featuremapWithCoef, target) 
        mixedFeature_2 = self.GraphNeuralNetwork(mixedSemanticFeature_2) 
        mixedOutput_2 = torch.tanh(self.fc(torch.cat((mixedFeature_2.view(batchSize * self.classNum, -1),
                                                      mixedSemanticFeature_2.view(-1, self.imageFeatureDim)),1)))
        mixedOutput_2 = mixedOutput_2.contiguous().view(batchSize, self.classNum, self.outputDim)
        mixedResult_2 = self.classifiers(mixedOutput_2)                          # (batchSize, classNum)

        return result, semanticFeature, mixedResult_1, mixedTarget_1, mixedResult_2, mixedTarget_2

    def mixupInstance_featuremap(self, F, T):
        """
        F: (batchSize, imgSize, imgSize, classNum, imgFeatureDim)
        T: (batchSize, classNum)
        """

        flipF, flipT = torch.flip(F, dims=[0]), torch.flip(T, dims=[0])

        matrix = torch.ones_like(T).cuda()
        matrix[(T == 0) & (flipT == 1)] = self.alpha
        mixedTarget = matrix * T + (1-matrix) * flipT

        matrix = matrix.unsqueeze(1).unsqueeze(1).unsqueeze(-1).repeat(1, F.size(1), F.size(1), 1, F.size(-1))
        mixedSemanticFeature = matrix * F + (1-matrix) * torch.flip(F, dims=[0])
        mixedSemanticFeature = torch.sum(torch.sum(mixedSemanticFeature, 1), 1)

        return mixedSemanticFeature, mixedTarget

    def mixupPrototype_featuremap(self, F, T):
        """
        F: (batchSize, imgSize, imgSize, classNum, imgFeatureDim)
        T: (batchSize, classNum)
        """

        halfSize, C = F.size(1) // 2, torch.sum(F, dim=-1)
        index0 = torch.argmax(torch.max(C, dim=2)[0], dim=1)
        index1 = torch.argmax(torch.max(C, dim=1)[0], dim=1)
        index = 2 * (index0 >= halfSize) + (index1 >= halfSize)

        index = (index - torch.randint(1, 4, index.size()).cuda()) % 4
        index = index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F.size(1), F.size(1), F.size(-1))

        prototype = self.prototype_featuremap.unsqueeze(0).repeat(F.size(0), 1, 1, 1, 1, 1)
        prototype = (index == 0) * prototype[:, :, 0].squeeze() + \
                    (index == 1) * prototype[:, :, 1].squeeze() + \
                    (index == 2) * prototype[:, :, 2].squeeze() + \
                    (index == 3) * prototype[:, :, 3].squeeze()
        prototype = prototype.transpose(1, 2).transpose(2, 3)                       # (batchSize, classNum, imgSize, imgSize, imgFeatureDim)

        matrix = torch.rand(T.size()).cuda()                                        # (batchSize, classNum)
        matrix = matrix * (T == 0)
        matrix = torch.zeros_like(matrix).cuda().scatter_(1, torch.topk(matrix, self.topK, dim=1)[1], 1)
      

        mixedTarget = (1-matrix) * T + matrix * (1-self.beta)
        matrix = matrix.unsqueeze(1).unsqueeze(1).unsqueeze(-1).repeat(1, F.size(1), F.size(1), 1, F.size(-1))
        mixedSemanticFeature = self.beta * F + (1-self.beta) * prototype
        mixedSemanticFeature = matrix * mixedSemanticFeature + (1-matrix) * F
        mixedSemanticFeature = torch.sum(torch.sum(mixedSemanticFeature, 1), 1)

        return mixedSemanticFeature, mixedTarget

    def computePrototype_featuremap(self, model, train_loader, args):

        model.eval()

        features = None
        for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(train_loader):

            input, target, groundTruth = input.cuda(), target.float().cuda(), groundTruth.cuda()

            with torch.no_grad():
                semanticFeature, featuremapWithCoef, Coefficient = model(input, onlyFeature=True)
                featuremapWithCoef, Coefficient = featuremapWithCoef.cpu(), Coefficient.cpu()

                if features is None:
                    features = [[torch.zeros(1, featuremapWithCoef.size(1), featuremapWithCoef.size(1), featuremapWithCoef.size(-1)) for _index in range(4)] for _class in range(args.classNum)]

                halfSize = Coefficient.size(1) // 2
                for _class in range(args.classNum):
                    for _batch in range(target.size(0)):

                        if target[_batch, _class] != 1:
                            continue

                        C = torch.sum(featuremapWithCoef[_batch, :, :, _class], dim=-1) 
                        index0 = torch.argmax(torch.max(C, dim=1)[0])
                        index1 = torch.argmax(torch.max(C, dim=0)[0])
                       
                        index = 2 * (index0 >= halfSize) + (index1 >= halfSize)
                        if features[_class][index].size(0) <= 100:
                            features[_class][index] = torch.cat((features[_class][index], featuremapWithCoef[_batch, :, :, _class].clone().unsqueeze(dim=0)), dim=0)

        prototypes = []
        for _class in range(args.classNum):

            _feature = [features[_class][i][1:] for i in range(4) if features[_class][i].size(0) > 1]
            avgPrototype = torch.cat(_feature, dim=0)
            avgPrototype = torch.mean(avgPrototype, dim=0).squeeze().cuda()

            for _index in range(4):
                if features[_class][_index].size(0) == 1:
                    features[_class][_index] = avgPrototype.clone()
                    continue
                features[_class][_index] = torch.mean(features[_class][_index], dim=0).squeeze().cuda()
             
            prototypes.append(torch.stack(features[_class], dim=0))
       
        model.prototype_featuremap = torch.stack(prototypes, dim=0)

    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)

    def load_matrix(self, mat):
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix, _out_matrix = nn.Parameter(torch.from_numpy(_in_matrix), requires_grad=False), nn.Parameter(torch.from_numpy(_out_matrix), requires_grad=False)
        return _in_matrix, _out_matrix

# =============================================================================
# Help Functions
# =============================================================================