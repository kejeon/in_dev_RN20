import torch
import torch.nn as nn
import torch.nn.functional as F

class Activate(nn.Module):
    def __init__(self, a_bit, quantize=True):
        super(Activate, self).__init__()
        self.abit = a_bit
        self.acti = nn.ReLU()
        self.quantize = quantize
        # assert self.abit <= 8 or self.abit == 32

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if self.quantize:
            x = qfn.apply(x, self.abit)
        return x

class qfn(torch.autograd.Function):
    @staticmethod                                                                   
    def forward(ctx, input, k):                                                     
        n = float(2**k - 1)
        out = torch.round(input * n) / n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class qfn_new(torch.autograd.Function):
    @staticmethod                                                                   
    def forward(ctx, input, k):                                                     
        n = float(k - 1)
        out = torch.round(input * n) / n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

""" Quantize weight"""

class weight_quantize_fn(nn.Module):
    def __init__(self, wbit):                                                   # bit_list: list to quantize (e.g. weight)
        super(weight_quantize_fn, self).__init__()
        # self.bit_list = bit_list
        # self.wbit = self.bit_list
        self.wbit = wbit
        # assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:                                                         # 32-bit: w_q_int, S 수정
            E = torch.mean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))
            weight_q = weight * E
        else:
            weight = torch.tanh(x)
            max_val = torch.max(torch.abs(weight))
            S = max_val / float(2**self.wbit - 1)
            num_levels = 2**(self.wbit + 1) - 1

            weight = weight / (2 * max_val) + 0.5                                   # w'

            weight_q_int = qfn_new.apply(weight, num_levels)                             # w_Q': [0, 1]

            weight_q = 2 * weight_q_int - 1                                         # w_Q: [-1, +1]
            weight_q = weight_q * max_val

        return weight_q, weight_q_int*(num_levels - 1), S
    

""" Quantized convolution """

class Conv2d_Q(nn.Conv2d):
    def __init__(self, w_bit, in_planes, planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d_Q, self).__init__(in_planes, planes, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)
        # fp weight given. warm start through post training quantization
        # 1. initialize with the inputted fp weight
        # self.scaling_factor = 0
        # self.w_q = self.weight
        # self.w_q_int = self.weight

        self.w_q, self.w_q_int, self.scaling_factor = self.quantize_fn(self.weight)

    def forward(self, input):
        # self.w_q = nn.Parameter(self.quantize_fn(self.weight)[0])
        # self.w_q_int = nn.Parameter(self.quantize_fn(self.weight)[1])
        # self.scaling_factor = nn.Parameter(self.quantize_fn(self.weight)[2])

        self.w_q, self.w_q_int, self.scaling_factor = self.quantize_fn(self.weight)

        return F.conv2d(input, self.w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)   # output: tensor

""" switchable batch normalization """

class BatchNorm2d_Q(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, a_bit, w_bit, num_features):                                 # a_bit != w_bit 수정
        super(BatchNorm2d_Q, self).__init__()
        self.abit = a_bit
        self.wbit = w_bit
        self.bn_dict = nn.ModuleDict()
        self.bn_dict[str(a_bit)] = nn.BatchNorm2d(num_features, eps=1e-4)           # w_bit -> a_bit  수정

        """
        self.abit = self.w_bit
        self.wbit = self.w_bit
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')
        """

    def forward(self, x):
        x = self.bn_dict[str(self.abit)](x)
        return x
    
""" Quantized linear """

class Linear_Q(nn.Linear):
    def __init__(self, w_bit, in_features, out_features, bias=True):
        super(Linear_Q, self).__init__(in_features, out_features, bias=bias)
        self.w_bit = w_bit
        self.w_quantize_fn = weight_quantize_fn(self.w_bit)                         # weight qnt function
        self.b_quantize_fn = weight_quantize_fn(self.w_bit)                         # bias qnt function

        # fp weight given. warm start through post training quantization
        # 1. initialize with the inputted fp weight
        # self.scaling_factor = 0
        # self.w_q = self.weight
        # self.w_q_int = self.weight

        self.w_q, self.w_q_int, self.scaling_factor = self.w_quantize_fn(self.weight)
        self.b_q, self.b_q_int, _ = self.b_quantize_fn(self.weight)

    def forward(self, input, order=None):
        # self.w_q = nn.Parameter(self.quantize_fn(self.weight)[0])
        # self.w_q_int = nn.Parameter(self.quantize_fn(self.weight)[1])
        # self.scaling_factor = nn.Parameter(self.quantize_fn(self.weight)[2])

        self.w_q, self.w_q_int, self.scaling_factor = self.w_quantize_fn(self.weight)
        self.b_q, self.b_q_int, _ = self.b_quantize_fn(self.bias)
        
        return F.linear(input, self.w_q, self.b_q)

if __name__ == "__main__":
    import random
    from torch.nn import Conv2d

    random_seed = 0
    torch.manual_seed(random_seed)

    conv_q = Conv2d_Q(w_bit = 1, in_planes=1, planes=1, kernel_size=3)

    # print(conv_q.weight)
    # print(conv_q.w_q_int)
    # print(conv_q.scaling_factor)
    # print(conv_q.w_q)
    conv_f = Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
    # conv_f.weight = nn.Parameter(conv_q.w_q)
    conv_f.weight = conv_q.weight

    print(conv_q.w_q_int)
    print(conv_q.w_q)
    print(conv_f.weight)

    x = torch.randn(1, 1, 8, 8)
    # print(x)

    out_q = conv_q(x)
    out_f = conv_f(x)

    print(out_f)
    print(out_q)