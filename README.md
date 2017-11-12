# CapsuleNet-Gluon
Implemention of Capsule Net from the paper Dynamic Routing Between Capsules Edit
Add topics

# Run

1. pip install --pre mxnet-cu80 -i https://pypi.douban.com/simple --user
2. pip install tqdm
3. python main.py

# Results

Results to be added

# Issues
I use `SGD` to try Capsule Net, if use `Adam` as described in the original paper, I got very low test accuracy.

I find if use large train batch_size, the results is also unaceptable. I haven't figure it out.

Any PR is welcomed.

# Other Implementations

- Kaggle (this version as self-contained notebook):
  - [MNIST Dataset](https://www.kaggle.com/kmader/capsulenet-on-mnist) running on the standard MNIST and predicting for test data
  - [MNIST Fashion](https://www.kaggle.com/kmader/capsulenet-on-fashion-mnist) running on the more challenging Fashion images.
- TensorFlow:
  - [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git)   
  Very good implementation. I referred to this repository in my code.
  - [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)   
  I referred to the use of tf.scan when optimizing my CapsuleLayer.
  - [LaoDar/tf_CapsNet_simple](https://github.com/LaoDar/tf_CapsNet_simple)

- PyTorch:
  - [tonysy/CapsuleNet-PyTorch](https://github.com/tonysy/CapsuleNet-PyTorch.git)
  - [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
  - [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
  - [andreaazzini/capsnet.pytorch](https://github.com/andreaazzini/capsnet.pytorch.git)
  - [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)
  
- MXNet:
  - [AaronLeong/CapsNet_Mxnet](https://github.com/AaronLeong/CapsNet_Mxnet)
  
- Lasagne (Theano):
  - [DeniskaMazur/CapsNet-Lasagne](https://github.com/DeniskaMazur/CapsNet-Lasagne)

- Chainer:
  - [soskek/dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)
