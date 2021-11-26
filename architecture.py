import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from myutils import hypersphere_random_sampler

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
	one-hidden-layer fully-connected with relu and convenient init
'''
class oneHL(nn.Module):

    def __init__(self, n_hidden, input_size, norm=None):

        super(oneHL, self).__init__()

        self.h = n_hidden
        self.d = input_size
        self.weight = nn.Parameter( hypersphere_random_sampler( n_hidden, input_size, device='cpu'))
        self.coeff = nn.Parameter( torch.cat( [torch.ones( 1, n_hidden // 2), -torch.ones( 1, n_hidden // 2)], dim=1))

        if norm=='mf':
            self.norm = float( n_hidden)
        elif norm=='ntk':
            self.norm = math.sqrt( float(n_hidden))
        else:
            self.norm = 1.0

    # Forward pass
    def forward(self, x):
        out = F.linear( x.reshape(-1, self.d), self.weight, bias=None)
        out = F.relu( out)
        out = F.linear( out, self.coeff, bias=None)/ self.norm
        return out

''' 
	one-hidden-layer convolutional with relu + avg pooling
'''
class Cnn1dReLUAvg(nn.Module):

    def __init__(self, n_hidden, filter_size, bias=False, pbc=True, stride=1, norm=None, smooth=1):

        super(Cnn1dReLUAvg, self).__init__()

        self.h = n_hidden
        self.fs = filter_size
        self.bias = bias
        self.smooth = smooth

        if pbc:
            self.cnn1 = Conv1dPBC_N(1, n_hidden, filter_size=filter_size, bias=bias)
        else:
            self.cnn1 = Conv1d_N(1, n_hidden, filter_size=filter_size, stride=stride, bias=bias)
        self.pool1 = AvgPool1d()
        self.fc1 = Linear_N(n_hidden, 1, bias=False)

        if norm=='mf':
            self.norm = float( n_hidden)
        elif norm=='ntk':
            self.norm = math.sqrt( float(n_hidden))
        else:
            self.norm = 1.0

    # Forward pass
    def forward(self, x):
        out = self.cnn1( x/math.sqrt( float( self.fs)))
        if self.smooth == 0:
            out = torch.heaviside( out, torch.zeros_like(out))
        else:
            out = torch.pow( F.relu( out), self.smooth)
        out = self.pool1( out)
        out = out.view( out.size(0), -1)
        out = self.fc1( out)/ self.norm
        return out
