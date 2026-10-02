# imports
import torch.nn as nn
import torch.nn.functional as F

# linear operation super
class Linear(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, use_bias = True):
        # initialized with weights and biases
        super().__init__(super_in_dim, super_out_dim, bias = use_bias)
        
        # notice that super_in_dim and super_out_dim are the largest values possible
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # the current configuration
        self.sample_in_dim = None
        self.sample_out_dim = None

        # initialize the parameters
        self.samples = {}
        self.use_bias = use_bias
        nn.init.kaiming_uniform_(self.weight)
        if (use_bias):
            nn.init.constant_(self.bias, 0.)


    def set_sample_config(self, sample_in_dim = None, sample_out_dim = None, is_identity_layer = False):
        if (is_identity_layer):
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        sample_weight = self.weight[:, :sample_in_dim]
        sample_weight = sample_weight[:sample_out_dim, :]
        self.samples['w'] = sample_weight
        if (self.use_bias):
            self.samples['b'] = self.bias[:sample_out_dim]


    def forward(self, x):
        # if this layer is skipped, do nothing
        if (self.is_identity_layer): return x
        w = self.weight[:self.sample_out_dim, :self.sample_in_dim]
        if (self.use_bias):
            res = F.linear(x, w, self.bias[:self.sample_out_dim])
        else:
            res = F.linear(x, w)
        return res


    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        if (self.is_identity_layer): return ""
        indent_str = "\t" * indent
        b_list = list(self.samples['b'].shape)
        return f"{indent_str}[W: {list(self.samples['w'].shape)}, b: {b_list[0] if len(b_list) == 1 else b_list}]"



