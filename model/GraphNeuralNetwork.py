import os
import sys
import numpy as np

import torch
import torch.nn as nn

class GatedGNN(nn.Module):

    def __init__(self, inputDim, timeStep, inMatrix, outMatrix):

        super(GatedGNN, self).__init__()

        self.inputDim, self.timeStep, self.inMatrix, self.outMatrix = inputDim, timeStep, inMatrix, outMatrix

        self.fc_1_w, self.fc_1_u = nn.Linear(2 * inputDim, inputDim), nn.Linear(inputDim, inputDim)
        self.fc_2_w, self.fc_2_u = nn.Linear(2 * inputDim, inputDim), nn.Linear(inputDim, inputDim)
        self.fc_3_w, self.fc_3_u = nn.Linear(2 * inputDim, inputDim), nn.Linear(inputDim, inputDim)

    def forward(self, input):
        """
        Shape of input : (BatchSize, classNum, inputDim)
        Shape of adjMatrix : (classNum, BatchSize, BatchSize)
        """

        batchSize, nodeNum = input.size()[0], self.inMatrix.size()[0]

        allNodes = input                                                             # BatchSize * nodeNum * inputDim
        inMatrix = self.inMatrix.repeat(batchSize, 1).view(batchSize, nodeNum, -1)   # BatchSize * nodeNum * nodeNum
        outMatrix = self.outMatrix.repeat(batchSize, 1).view(batchSize, nodeNum, -1) # BatchSize * nodeNum * nodeNum

        for time in range(self.timeStep):
  
            # See eq(8) for more details
            a_c = torch.cat((torch.bmm(inMatrix, allNodes),
                             torch.bmm(outMatrix, allNodes)), 2)                     # BatchSize * nodeNum * (2 * inputDim)
            a_c = a_c.contiguous().view(batchSize * nodeNum, -1)                     # (BatchSize * nodeNum) * (2 * inputDim)

            flatten_allNodes = allNodes.view(batchSize * nodeNum, -1)                # (BatchSize * nodeNum) * inputDim

            # See 1th row of eq(3) for more details
            z_c = torch.sigmoid(self.fc_1_w(a_c) + self.fc_1_u(flatten_allNodes))    # (BatchSize * nodeNum) * inputDim 

            # See 2th row of eq(3) for more details
            r_c = torch.sigmoid(self.fc_2_w(a_c) + self.fc_2_u(flatten_allNodes))    # (BatchSize * nodeNum) * inputDim
           
            # See 3th row of eq(3) for more details
            h_c = torch.tanh(self.fc_3_w(a_c) + self.fc_3_u(r_c * flatten_allNodes)) # (BatchSize * nodeNum) * inputDim

            # See 4th row of eq(3) for more details
            flatten_allNodes = (1 - z_c) * flatten_allNodes + z_c * h_c              # (BatchSize * nodeNum) * inputDim

            allNodes = flatten_allNodes.view(batchSize, nodeNum, -1)                 # BatchSize * nodeNum * inputDim

        return allNodes