import torch
import torch.nn as nn
import torch.nn.functional as F
from model.q_lib import Activate, BatchNorm2d_Q, Linear_Q, Conv2d_Q


class ResNet_Q(nn.Module):
    def __init__(self, a_bit, w_bit, block, num_layers=3, num_classes=10):
        super(ResNet_Q, self).__init__()
        
        self.in_planes = 16 # Resnet
        self.num_layers = num_layers

        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act = Activate(self.a_bit)
        # self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                           stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d_Q(self.a_bit, self.w_bit, self.in_planes)

        # 32x32x16
        self.layers_2n = self._make_layer(block, 16, 16, stride=1)
        # 16x16x32
        self.layers_4n = self._make_layer(block, 16, 32, stride=2)
        # 8x8x64
        self.layers_6n = self._make_layer(block, 32, 64, stride=2)

        """
        self.layers = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d_Q(self.w_bit, self.in_planes),
            Activate(self.a_bit),
            # nn.ReLU(),
            
            *self._make_layer(block, 16, num_blocks[0], stride=1), ## 현재 이거 수정함
            *self._make_layer(block, 32, num_blocks[1], stride=2),
            *self._make_layer(block, 64, num_blocks[2], stride=2),
        )
        """

        # mask_prune(self.layers)
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        
        # self.fc = nn.Linear(64, num_classes)
        self.fc = Linear_Q(self.w_bit, 64, num_classes)

    def _make_layer(self, block, in_planes, planes, stride):
        if stride == 2:
            down_sample = True
        else:
            down_sample = False

        layers = nn.ModuleList(
            [block(self.a_bit, self.w_bit, in_planes, planes, stride, down_sample)]
        )

        for _ in range(1, self.num_layers):
            layers.append(block(self.a_bit, self.w_bit, planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.layers_2n(x)
        x = self.layers_4n(x)
        x = self.layers_6n(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
class BasicBlock_Q(nn.Module):
    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1, down_sample=False):
        super(BasicBlock_Q, self).__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit

        self.act1 = Activate(self.a_bit)
        self.act2 = Activate(self.a_bit)

        self.conv1 = Conv2d_Q(self.w_bit, in_planes, planes, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d_Q(self.a_bit, self.w_bit, planes)
        self.conv2 = Conv2d_Q(self.w_bit, planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d_Q(self.a_bit, self.w_bit, planes)

        # dropout: p=0.2
        self.dropout = nn.Dropout(0.2)
        
        self.stride = stride

        if down_sample:
            self.down_sample = IdentityPadding(in_planes, planes, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x

        out = self.act1(self.bn1(self.conv1(x)))

        # dropout: p=0.2
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample != None:
            shortcut = self.down_sample(x)
        out += shortcut

        out = self.act2(out)

        return out

class IdentityPadding(nn.Module):
  def __init__(self, in_planes, planes, stride):
    super(IdentityPadding, self).__init__()
    self.pooling = nn.MaxPool2d(1, stride=stride) 
    self.add_channels = planes - in_planes

  def forward(self, x):
    out = nn.functional.pad(x, (0, 0, 0, 0, 0, self.add_channels))
    out = self.pooling(out)
    return out
  
def ResNet20_Q(a_bit, w_bit):
    return ResNet_Q(a_bit, w_bit, BasicBlock_Q, num_layers=3)