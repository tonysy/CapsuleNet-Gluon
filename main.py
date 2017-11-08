from __future__ import print_function
import sys
import os
import numpy as np 
import mxnet as mx
from mxnet.gluon import nn

from data_dowloader import download_and_create_data


if __name__ == '__main__':
    #########################################################
    ###          Load Dataset and check shape             ###
    #########################################################
    trainX, trainY, testX, testY = download_and_create_data()

    print('Training Images Shape:'.format(trainX.shape))
    print('Training Labels Shape:'.format(trainY.shape))

    print('Test Images Shape:'.format(testX.shape))
    print('Test Labels Shape:'.format(testY.shape))


