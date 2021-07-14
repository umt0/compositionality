import sys
import copy
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math

def is_valid(locations, length, size):

    '''input a sorted list of positions in [0,length), 
       check if it can host objects of size "size" 
       with exclusion'''

    previous = locations[-1] - length
    for l in locations:
        if l - previous < size:
            return False
        previous = l

    return True

class clam:

    '''draw normalised random vector "pat" with label 
       "cl" depending on the alignment with DIRECTOR '''

    def __init__(self, director):
        self.n = director

    def draw(self):

        pat = torch.randn( self.n.size(0))
        pat /= pat.norm()
        cl = 2*(torch.dot( pat, self.n).item() > 0) - 1

        return pat, cl

class pair:

    '''draw a vector "pat" from one of two PATTERNS
       with class "cl" -1 for the first and +1 for the second  '''

    def __init__(self, patterns):
        self.p = patterns

    def draw(self):

        index = random.randrange(2)
        pat = self.p[index]
        cl = 2*index - 1

        return pat, cl

class noisy:

    '''draw a vector "pat" from one of - and + PATTERN	
       with class "cl" -1 for the first and +1 for the second
       and rotate in a random direction by a random gaussian 
       angle with variance (pi * EPS) ** 2 '''

    def __init__(self, pattern, eps):
        self.p = pattern
        self.e = eps

    def draw(self):

        index = random.randrange(2)
        cl = 2*index - 1
        pat = cl * self.p
        size = self.p.size(0)

        angle = torch.randn(1) * math.pi * self.e
        randir = torch.randn( size)
        randax = randir - torch.dot( pat, randir) * pat
        randax /= randax.norm()
        pat = pat * torch.cos( angle) + randax * torch.sin( angle)
        
        return pat, cl

''' 
1d input with NUMBER patterns drawn according to MODE
'''
def binary_counter_1d( set_size, image_size, pattern_size, number, mode, director=None, patterns=None, noise=None, exclusion=True, centered=False):

    if mode == 'clam':
        assert director is not None, 'mode clam requires director'
        assert len(director.size()) == 1, 'expect rank-1 tensor as dir.'
        assert director.size(0) == pattern_size, 'dir. must have same size as patt.'
        generator = clam( director)

    elif mode == 'pair':
        assert patterns is not None, 'mode pair requires target patt.'
        assert len(patterns.size()) == 2, 'expect rank-2 tensor as target patt.'
        assert patterns.size(0) == 2, 'expect two patt.'
        assert patterns.size(1) == pattern_size, 'target patt. must have same size as patt.'
        generator = pair( patterns)

    elif mode == 'noisy':
        assert director is not None, 'mode noisy requires director'
        assert len(director.size()) == 1, 'expect rank-1 tensor as dir.'
        assert director.size(0) == pattern_size, 'dir. must have same size as patt.'
        assert noise is not None, 'mode noisy requires noise'
        generator = noisy( director, noise)

    else:
        raise AssertionError('mode -' + mode + '- not implemented')

    assert pattern_size * number <= image_size, 'image too crowded'

    # build a list of available places
    if centered:
        assert image_size % pattern_size == 0, 'im.size must be multiple of patt.size'
        available_spots = [i for i in range(0, image_size, pattern_size)]
    else:
        available_spots = [i for i in range(0, image_size)]

    # initialise data (x) and labels (y)
    x = torch.zeros(set_size, 1, image_size)
    y = torch.zeros(set_size)    

    for mu in range(set_size):

        # set label to -1 and change it if class-(1) patterns are more numerous
        y[mu] = -1
        label = 0

        if exclusion:

            valid = False
            while not valid:

                locations = random.sample(available_spots, number)
                locations.sort()
                valid = is_valid( locations, image_size, pattern_size)

        else:

            locations = random.sample(available_spots, number)
            locations.sort()

        for l in locations:

            pat, cl = generator.draw()
            for i in range(pattern_size):
                x[mu, 0, (l + i) % image_size] += pat[i]
            label += cl

        if label > 0:
            y[mu] = +1

    # reshape data and labels
    x = x.reshape(set_size, 1, image_size)
    y = y.reshape(set_size, 1)
    
    return x, y
