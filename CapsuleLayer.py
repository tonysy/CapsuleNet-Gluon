from collections import OrderedDict
from mxnet.gluon import nn
import mxnet.ndarray as nd

class CapsuleConv(nn.Block):
    def __init__(self,dim_vcetor, out_channels, kernel_size, 
                        strides=1,padding=0):
        super(CapsuleConv, self).__init__()
        
        self.capsules_index = ['dim_'+str(i) for i in range(dim_vcetor)]
        for idx in self.capsules_index:
            setattr(self, idx, nn.Conv2D(out_channels, 
                    kernel_size=kernel_size, strides=strides,
                    padding=padding))

    def squash(self, tensor):
        """Batch Squashing Function
        
        Args:
            tensor : 5-D, (batch_size, num_channel,  height, width, dim_vector)
            
        Return:
            tesnor_squached : 5-D, (batch_size, num_channel, height, width, dim_vector)
        """
        tensor_l2norm = (tensor**2).sum(axis=-1).expand_dims(axis=-1)
        scale_factor = tensor_l2norm / (1 + tensor_l2norm)
        tensor_squashed = tensor * (scale_factor / tensor_l2norm**0.5)

        return tensor_squashed

    def forward(self, X):
                    
        outputs = [getattr(self,idx)(X).expand_dims(axis=-1) for idx in self.capsules_index]

        outputs_cat = nd.concatenate(outputs, axis=4)
        outputs_squashed = self.squash(outputs_cat)
        return outputs_squashed

class CapsuleDense(nn.Block):
    def __init__(self, dim_vector, dim_input_vector, out_channels,
                        num_routing_iter=1):
        super(CapsuleDense, self).__init__()

        self.dim_vector = dim_vector
        self.dim_input_vector = dim_input_vector
        self.out_channels = out_channels
        self.num_routing_iter = num_routing_iter
        self.routing_weight_initial = True
    
    def squash(self, tensor):
        """Batch Squashing Function
        
        Args:
            tensor : 5-D, (batch_size, num_channel,  height, width, dim_vector)
            
        Return:
            tesnor_squached : 5-D, (batch_size, num_channel, height, width, dim_vector)
        """
        tensor_l2norm = (tensor**2).sum(axis=-1).expand_dims(axis=-1)
        scale_factor = tensor_l2norm / (1 + tensor_l2norm)
        tensor_squashed = tensor * (scale_factor / tensor_l2norm**0.5)

        return tensor_squashed

    def forward(self, X):
        # (batch_size, num_channel_prev, h, w, dim_vector)
        # -->(batch_size,num_capsule_prev,1,1,dim_vector)
        X = X.reshape((0, -1, 1, 1, 0))

    
        self.num_capsules_prev = X.shape[1]
        self.batch_size = X.shape[0]
        # (batch_size,num_capsule_prev,out_channels,1,dim_vector)
        X_tile = nd.tile(X, reps=(1,1,self.out_channels,1,1))

        if self.routing_weight_initial:
            self.routing_weight = nd.random_normal(shape=(1,
                self.num_capsules_prev,self.out_channels,
                self.dim_input_vector, self.dim_vector), name='routing_weight')
            self.routing_weight_initial = False
        # (batch_size,num_capsule_prev,out_channels,dim_input_vector,dim_vector)
        # (64, 1152, 10, 8, 16)
        W_tile = nd.tile(self.routing_weight, reps=(self.batch_size,1,1,1,1))
        linear_combination_3d = nd.batch_dot(
                X_tile.reshape((-1, X_tile.shape[-2], X_tile.shape[-1])), 
                W_tile.reshape((-1, W_tile.shape[-2], W_tile.shape[-1])))
        # (64, 1152, 10, 1, 16)
        linear_combination = linear_combination_3d.reshape((self.batch_size,
                                self.num_capsules_prev, self.out_channels,
                                1, self.dim_vector))

        # b_ij (1, 1152, 10, 1, 1)
        priors = nd.zeros((1, self.num_capsules_prev,self.out_channels,1,1))

        ############################################################################
        ##                                Rounting                                ##
        ############################################################################
        for iter_index in range(self.num_routing_iter):
            # NOTE: RoutingAlgorithm-line 4
            # b_ij (1, 1152, 10, 1, 1)
            softmax_prior = nd.softmax(priors, axis=2) # on num_capsule dimension
            # NOTE: RoutingAlgorithm-line 5
            # (64, 1152, 10, 1, 16)
            # output = torch.mul(softmax_prior, linear_combination)
            output =  softmax_prior * linear_combination 

            # (64, 1, 10, 1, 16)
            output_sum = output.sum(axis=1, keepdims=True) # s_J

            # NOTE: RoutingAlgorithm-line 6
            # (64, 1, 10, 1, 16)
            output_squashed = self.squash(output_sum) # v_J

            # NOTE: RoutingAlgorithm-line 7
            # (64, 1152, 10, 1, 16)
            output_tile = nd.tile(output_squashed, reps=(1,self.num_capsules_prev,1,1,1))
            # (64, 1152, 10, 1, 16) x (64, 1152, 10, 1, 16) (transpose on last two axis) 
            # ==> (64, 1152, 10, 1, 1)
            U_times_v = nd.batch_dot(linear_combination.reshape((-1, 1, self.dim_vector)),
                                     output_tile.reshape((-1, 1, self.dim_vector)),
                                     transpose_b =True)
            U_times_v = U_times_v.reshape((self.batch_size, self.num_capsules_prev,
                                        self.out_channels, 1, 1))

            priors = priors + U_times_v.sum(axis=0).expand_dims(axis=0)
    
        return output_squashed # v_J

def main():
    a = nd.random.uniform(shape=(64,32,6,6,8))    
    model = CapsuleDense(16, 8, 10, num_routing_iter=3)
    model.initialize()
    y = model(a)
    print(y.shape)
if __name__ == '__main__':
    main()
    