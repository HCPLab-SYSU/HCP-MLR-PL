import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Element_Wise_Layer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):

        super(Element_Wise_Layer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)

        self.initParameters()

    def forward(self, input):
        """
        Shape of input : BatchSize * classNum * featureDim
        """

        output = torch.sum(input * self.weight, 2) # BatchSize * classNum
        if self.bias is not None:
            output = output + self.bias

        return output

    def initParameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))

        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)

        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)
