import sys
import copy
import functools
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math

import time

from dataset import binary_counter_1d
from architecture import Cnn1dReLUAvg
from gradientflow import gradientflow_backprop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

class HingeLoss_a(torch.nn.Module):

    def __init__(self, alpha):
        super(HingeLoss_a, self).__init__()

        self.alpha = alpha

    def forward(self, output, target):
        hinge_loss = 1 - self.alpha*output*target
        return F.relu(hinge_loss).mean(dim=1) / (self.alpha * self.alpha)

class ScalarTestError(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        error = ( 1 - torch.sign( (output*target)))*0.5
        return  error.mean()


parser = argparse.ArgumentParser(description='Train a CNN on shape models')

'''
	DATASET ARGS
'''
parser.add_argument('--trainsize', metavar='P', type=int, help='size of the training set')
parser.add_argument('--imagesize', metavar='IM.SIZE', type=int, help='size of the training images')
parser.add_argument('--patternsize', metavar='PAT.SIZE', type=int, help='size of the patterns', default=None)
parser.add_argument('--number', type=int, help='patterns per image', default=None)
parser.add_argument('--mode', type=str, help='noisy, clam', default='noisy')
parser.add_argument('--noise', type=float, default=None)
parser.add_argument('--seed', type=int, help='seed for the patterns', default=None)
parser.add_argument('--exclusion', action='store_true', default=False)
parser.add_argument('--centered', action='store_true', default=False)
'''
	ARCHITECTURE ARGS
'''
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--hidden', metavar='H', type=int, help='width of the network')
parser.add_argument('--filtersize', metavar='FS', type=int, help='filter size')
parser.add_argument('--stride', metavar='STR', type=int, help='stride', default=1)
parser.add_argument('--pbc', action='store_true', default=False)
parser.add_argument('--bias', action='store_true', default=False)
'''
	TRAINING ARGS
'''
parser.add_argument('--loss', type=str, help='hinge, logistic', default='hinge')
'''
	OUTPUT ARGS
'''
parser.add_argument('--testsize', type=int, help='size of the test set', default=2048)
parser.add_argument('--maxtime', type=float, help='maximum time in hours', default=23.5)
parser.add_argument('--savestep', type=int, help='frequency of saves in steps', default=100)
parser.add_argument('--minfrac', type=float, help='goal training loss', default=0.0)
parser.add_argument('--array', type=int, help='index for array runs', default=None)

args = parser.parse_args()

train_size = args.trainsize
test_size = args.testsize
image_size = args.imagesize
pattern_size = args.patternsize
number = args.number

torch.manual_seed(args.seed)
director = torch.randn( pattern_size)
director /= director.norm()
x_train, y_train = binary_counter_1d(train_size, image_size, pattern_size, number, args.mode, director=director, noise=args.noise, exclusion=args.exclusion, centered=args.centered)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test, y_test = binary_counter_1d(test_size, image_size, pattern_size, number, args.mode, director=director, noise=args.noise, exclusion=args.exclusion, centered=args.centered)
x_test = x_test.to(device)
y_test = y_test.to(device)
training_set = [x_train, y_train]

alpha = args.alpha
hidden = args.hidden
filter_size = args.filtersize
stride = args.stride
model = Cnn1dReLUAvg(hidden, filter_size, bias=args.bias, pbc=args.pbc, stride=stride).to(device)
model_init = copy.deepcopy(model)

if args.loss == 'hinge':
    loss_f = HingeLoss_a(alpha)
elif args.loss == 'logistic':
    loss_f = LogLoss_a(alpha)
else:
    raise ValueError('loss "' +args.loss + '" not implemented')
test_f = ScalarTestError()

dynamics = []
freq = args.savestep
max_time = args.maxtime
min_frac = args.minfrac

start_time = time.time()
stop = False

for state, internals in gradientflow_backprop(model, x_train, y_train, loss_f, subf0=True, max_dgrad=1e10, max_dout=1e-1/alpha):

    if alpha * alpha * state['loss'] < min_frac:

        with torch.no_grad():
            out = internals['f'](x_test)
        test = alpha * ( out - model_init(x_test))
        testerr = test_f(test, y_test).item()

        current = state
        current['test'] = testerr
        print(current)
        dynamics.append(current)
        break

    if time.time() - start_time > max_time * 3600:

        with torch.no_grad():
            out = internals['f'](x_test)
        test = alpha * ( out - model_init(x_test))
        testerr = test_f(test, y_test).item()

        current = state
        current['test'] = testerr
        print(current)
        dynamics.append(current)
	stop = True
        break

    if state['step'] % freq == 0:

        with torch.no_grad():
            out = internals['f'](x_test)
        test = alpha * ( out - model_init(x_test))
        testerr = test_f(test, y_test).item()

        current = state
        current['test'] = testerr
        print(current)
        dynamics.append(current)

model_train = internals['f']

filename = 'feature_binary'
filename += '_d' + str(args.imagesize)
filename += '_t' + str(args.patternsize)
filename += '_n' + str(args.number)
filename += '_' + str(args.mode)
if args.centered:
    filename += '_c'
if not args.exclusion:
    filename += 'o'

filename += '_s' + str(args.filtersize) + '-' + str(args.stride)
if args.bias:
    filename += '_b'
if args.pbc:
    filename += '_pbc'
filename += '_h' + str(hidden)
filename += '_a' + str(alpha)
filename += '_P' +str(train_size)

if args.array is not None:
    filename += '_' + str(args.array)

if stop:
    filename += '_stop'

torch.save({
            'args': args,
            'director': director,
            'model_train': model_train.state_dict(),
            'model_init': model_init.state_dict(),
            'training_set': training_set,
            'dynamics': dynamics,
	    'stopped': stop
           }, filename +'.pt')
