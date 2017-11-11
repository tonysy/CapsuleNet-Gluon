from mxnet import gluon
from mxnet import autograd
from mxnet import nd 
from mxnet import image
from mxnet.gluon import nn
import mxnet as mx
import numpy as np 

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()

    return ctx

def try_all_gpu():
    """Return all available GPUs, or [mx.cpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass

    if not ctx_list:
        ctx_list = [mx.cpu()]

    return ctx_list

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def concact_vectors_in_list(vec_list, axis):
    concat_vec = vec_list[0]
    for i in range(1, len(vec_list)):
        concat_vec = nd.concat(concat_vec, vec_list[i], dim=axis)

    return concat_vec