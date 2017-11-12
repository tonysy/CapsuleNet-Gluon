from mxnet.gluon import nn
from mxnet.gluon.loss import Loss, L2Loss,  _apply_weighting
from mxnet import nd
from CapsuleLayer import CapsuleConv, CapsuleDense

class CapsuleNet(nn.HybridBlock):
    def __init__(self, *args, **kwargs):
        super(CapsuleNet, self).__init__(**kwargs)

        with self.name_scope():
            
            conv1 = nn.HybridSequential()
            conv1.add(
                # Conv1
                nn.Conv2D(256,kernel_size=9, strides=2, activation='relu')
            )
            
            primary = nn.HybridSequential()
            primary.add(
                CapsuleConv(dim_vector=8,out_channels=32,
                                    kernel_size=9, strides=2)
            )

            digit = nn.HybridSequential()
            digit.add(
                CapsuleDense(dim_vector=16, dim_input_vector=8,
                                            out_channels=10, num_routing_iter=3)
            )
            
            decoder_module = nn.HybridSequential()
            decoder_module.add(
                        nn.Dense(512, activation='relu'),
                        nn.Dense(1024, activation='relu'),
                        nn.Dense(784, activation='sigmoid'))

            self.net = nn.HybridSequential()
            self.net.add(conv1, primary, digit, decoder_module)
    
    def hybrid_forward(self, F, X, y=None):
        # import pdb; pdb.set_trace()
        X = self.net[0](X) # Conv1
        X = self.net[1](X) # Primary Capsule
        X = self.net[2](X) # Digital Capsule
        # import pdb ; pdb.set_trace()
        X = X.reshape((X.shape[0],X.shape[2], X.shape[4]))
        # get length of vector for margin loss calculation
        X_l2norm = nd.sqrt((X**2).sum(axis=-1))
        # import pdb ; pdb.set_trace()
        prob = nd.softmax(X_l2norm, axis=-1)

        if y is not None:
            max_len_indices = y
        else:
            
            max_len_indices = nd.argmax(prob,axis=-1)


        y_tile = nd.tile(y.expand_dims(axis=1), reps=(1, X.shape[-1]))
        batch_activated_capsules = nd.pick(X, y_tile, axis=1, keepdims=True)

        reconstrcutions = self.net[3](batch_activated_capsules)

        return  prob, X_l2norm, reconstrcutions

class CapsuleMarginLoss(Loss):
    """Calculates margin loss for CapsuleNet between output and label:

    .. math::
        

    Output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(CapsuleMarginLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, images, num_classes, labels, X_l2norm,
                    lambda_value = 0.5, sample_weight=None):
        self.num_classes = num_classes
        labels_onehot = nd.one_hot(labels, num_classes)
        first_term_base = F.square(nd.maximum(0.9-X_l2norm,0))
        second_term_base = F.square(nd.maximum(X_l2norm -0.1, 0))
        # import pdb; pdb.set_trace()
        margin_loss = labels_onehot * first_term_base + lambda_value * (1-labels_onehot) * second_term_base
        margin_loss = margin_loss.sum(axis=1) 

        loss = F.mean(margin_loss, axis=self._batch_axis, exclude=True) 
        loss = _apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
    