from __future__ import print_function
import sys
import os
import argparse
import datetime
import numpy as np 
import mxnet as mx
from tqdm import tqdm
from mxnet.gluon import nn
from mxnet.gluon.loss import L2Loss
from mxnet import nd, autograd, gluon, init
from CapsuleNet import CapsuleNet, CapsuleMarginLoss

mx.random.seed(1)
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='CapsuelNet Pytorch MINIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
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

def train(net,epochs, ctx, train_data,test_data,
            margin_loss, reconstructions_loss, 
            batch_size,scale_factor):
    num_classes = 10
    trainer = gluon.Trainer(
        net.collect_params(),'sgd', {'learning_rate': 0.05, 'wd': 5e-4})

    for epoch in range(epochs):
        train_loss = 0.0
        for batch_idx, (data, label) in tqdm(enumerate(train_data), total=len(train_data), ncols=70, leave=False, unit='b'):
            label = label.as_in_context(ctx)
            data = data.as_in_context(ctx)
            with autograd.record():
                prob, X_l2norm, reconstructions = net(data, label)
                loss1 = margin_loss(data, num_classes,  label, X_l2norm)
                loss2 = reconstructions_loss(reconstructions, data)
                loss = loss1 + scale_factor * loss2
                loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
        test_acc = test(test_data, net, ctx)
        print('Epoch:{}, TrainLoss:{:.5f}, TestAcc:{}'.format(epoch,train_loss / len(train_data),test_acc))
            
def test(data_iterator, net, ctx):
    acc = mx.metric.Accuracy()
    for i, (data, label) in tqdm(enumerate(data_iterator),total=len(data_iterator), ncols=70, leave=False, unit='b'):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        prob,_,_ = net(data,label)
        predictions = nd.argmax(prob, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

def main():
    args = parse_args()
    ctx = mx.gpu(0)
    scale_factor = 0.0005
    ##############################################################
    ###                    Load Dataset                        ###
    ##############################################################
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
    margin_loss = CapsuleMarginLoss()
    reconstructions_loss = L2Loss()
    # convert to static graph for speedup
    # capsule_net.hybridize()
    train(capsule_net, args.epochs,ctx,train_data,test_data, margin_loss,
            reconstructions_loss, args.batch_size, scale_factor)

if __name__ == '__main__':
    main()

