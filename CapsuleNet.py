from mxnet.gluon import nn
from mxnet.gluon import loss
from mxnet import nd
from CapsuleLayer import CapsuleConv, CapsuleDense

class CapsuleNet(nn.Block):
    def __init__(self, *args, **kwargs):
        super(CapsuleNet, self).__init__(**kwargs)

        with self.name_scope():
            
            conv1 = nn.Sequential()
            conv1.add(
                # Conv1
                nn.Conv2D(256,kernel_size=9, strides=2, activation='relu')
            )
            
            primary = nn.Sequential()
            primary.add(
                primary_capsules = CapsuleConv(dim_vector=8,out_channels=32,
                                    kernel_size=9, stride=2)
            )

            digit = nn.Sequential()
            digit.add(
                digit_capsules = CapsuleDense(dim_vector=16, dim_input_vector=8,
                                            out_channels=10, num_routing_iter=3)
            )
            
            decoder_module = nn.Sequential()
            decoder_module.add(
                        nn.Dense(512, activation='relu'),
                        nn.Dense(1024, activation='relu'),
                        nn.Dense(784, activation='sigmoid'))

            self.net = nn.Sequential()
            self.net.add(conv1, primary, digit, decoder_module)
    
    def forward(self, X, y=None):
        X = self.net[0](X) # Conv1
        X = self.net[1](X) # Primary Capsule
        X = self.net[2](X) # Digital Capsule

        X = X.reshape((X.shape(0),X.shape(2), X.shape(4)))
        # get length of vector for margin loss calculation
        X_l2norm = nd.sqrt(X**2).sum(axis=-1)


        if y is not None:
            max_len_indices = y
        else:
            prob = nd.softmax(X_l2norm, axis=-1)
            max_len_indices = nd.argmax(prob,axis=-1)
        
        batch_activated_capsules = X[range(X.shape[0]),max_len_indices.astype('int32')]

        reconstrcutions = self.net[3](batch_activated_capsules)

        return  X_l2norm, reconstrcutions

class CapsuleLoss(loss.Loss):
    """Calculates total loss for CapsuleNet between output and label:

    .. math::
        L = \\frac{1}{2}\\sum_i \\Vert {output}_i - {label}_i \\Vert^2.

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
        super(CapsuleLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.square(output - label)
        loss = _apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
    
    def hybrid_forward(self, F, images, labels, X_l2norm, reconstrcutions,
                            lambda_value=0.5, scalar_factor=0.0005)