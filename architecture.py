import torch
import torch.nn as nn
import torch.nn.functional as F
import math

''' Input of convolutional layers is given as a tensor of rank 2 + d,

        - input(conv) = Tensor[(batch_size)x(input_channels)x(image_length)^d]
        
    Input of fully-connected layers is given as a tensor of rank 2,
    
        - input(lin) = Tensor[(batch_size)x(input_features)]
    
   '''

class Linear_N(nn.Module):

    def __init__(self, in_features, out_features, bias=False):

        super(Linear_N, self).__init__()

        self.h = in_features
        self.weight = nn.Parameter( torch.randn( out_features, in_features))
        if bias:
            self.bias = nn.Parameter( torch.randn( out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class Conv1d_N(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, stride, bias=False):

        super(Conv1d_N, self).__init__()

        # initialise the filter with unit-variance Gaussian RV
        self.fs = filter_size
        self.str = stride
        self.filt = nn.Parameter( torch.randn( out_channels, in_channels, filter_size))
        if bias:
            self.bias = nn.Parameter( torch.randn( out_channels))
        else:
            self.register_parameter('bias', None)

    # return convolution of the input x with PBC with the filter
    def forward(self, x):
        return F.conv1d(x, self.filt, self.bias, stride=self.str)

class Conv1dPBC_N(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, bias=False):

        super(Conv1dPBC_N, self).__init__()

        # initialise the filter with unit-variance Gaussian RV
        self.fs = filter_size
        self.filt = nn.Parameter( torch.randn( out_channels, in_channels, filter_size))
        if bias:
            self.bias = nn.Parameter( torch.randn( out_channels))
        else:
            self.register_parameter('bias', None)

    # return convolution of the input x with PBC with the filter
    def forward(self, x):
        x_pbc = F.pad(x, (0, self.fs-1), mode='circular')
        return F.conv1d(x_pbc, self.filt, self.bias)

class AvgPool1d(nn.Module):

    def __init__(self):

        super(AvgPool1d, self).__init__()

    def forward(self, x):
        return F.avg_pool1d(x, x.size(2))

class MaxPool1d(nn.Module):
    
    def __init__(self):
        super(MaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, x.size(2))

''' 
	one-hidden-layer convolutional with relu + avg pooling
'''
class Cnn1dReLUAvg(nn.Module):

    def __init__(self, n_hidden, filter_size, bias=False, pbc=True, stride=1):

        super(Cnn1dReLUAvg, self).__init__()

        self.h = n_hidden
        self.fs = filter_size
        self.bias = bias

        if pbc:
            self.cnn1 = Conv1dPBC_N(1, n_hidden, filter_size=filter_size, bias=bias)
        else:
            self.cnn1 = Conv1d_N(1, n_hidden, filter_size=filter_size, stride=stride, bias=bias)
        self.pool1 = AvgPool1d()
        self.fc1 = Linear_N(n_hidden, 1, bias=False)

    # Forward pass
    def forward(self, x):
        pre_act = self.cnn1(x/math.sqrt( float( self.fs)))
        post_act = F.relu(pre_act)
        pooled = self.pool1(post_act)
        pooled = pooled.view(pooled.size(0), -1)
        out = self.fc1(pooled)/math.sqrt( float(self.h))
        return out
