from __future__ import print_function
import sys
# sys.path.insert()
import os
import argparse
import datetime
import numpy as np 
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, autograd, gluon, init
from CapsuleNet import CapsuleNet, CapsuleLoss
from utils import try_gpu, accuracy
mx.random.seed(1)
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='CapsuelNet Pytorch MINIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    args = parser.parse_args()
    
    return args

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

def train(net,epochs, ctx, train_data, capsule_loss, 
            batch_size):
    num_classes = 10
    trainer = gluon.Trainer(
        net.collect_params(), 'adam')
    prev_time = datetime.datetime.now()

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        for data, label in train_data:
            label = label.as_in_context(ctx)
            data = data.as_in_context(ctx)
            with autograd.record():
                prob, X_l2norm, reconstructions = net(data, label)
                loss = capsule_loss( data, num_classes,  label, X_l2norm, reconstructions)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(prob, label)
        # for log
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time %02d:%02d:%02d' % (h, m, s)

        epoch_str = ('Epoch %d. Loss: %f, Train acc %f, ' % (epoch, train_loss / len(train_data), train_acc / len(train_data)))

        prev_time = cur_time
        print(epoch_str + time_str)
            


def main():
    args = parse_args()
    ctx = try_gpu()

    #########################################################
    ###                    Load Dataset                   ###
    #########################################################
    train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True,
                                transform=transform),args.batch_size,
                                shuffle=True)
    test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False,
                                transform=transform),
                                args.test_batch_size,
                                shuffle=False)
    ##############################################################
    ##                  Load network and set optimizer          ##
    ##############################################################
    capsule_net = CapsuleNet()
    capsule_net.initialize(ctx=ctx, init=init.Xavier())
    capsule_loss = CapsuleLoss()
    train(capsule_net, args.epochs,ctx,train_data, capsule_loss,
            args.batch_size)

if __name__ == '__main__':
    main()
# from data_dowloader import download_and_create_data


# if __name__ == '__main__':
#     #########################################################
#     ###          Load Dataset and check shape             ###
#     #########################################################
#     trainX, trainY, testX, testY = download_and_create_data()

#     print('Training Images Shape:'.format(trainX.shape))
#     print('Training Labels Shape:'.format(trainY.shape))

#     print('Test Images Shape:'.format(testX.shape))
#     print('Test Labels Shape:'.format(testY.shape))


