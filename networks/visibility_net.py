from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import *


class VisibilityNet(nn.Module):
    def __init__(self, num_ch_enc):
        super(VisibilityNet, self).__init__()
        self.in_channels = num_ch_enc[-1]
        self.hid_channels = self.in_channels*2
        self.conv1 = ConvBlock(self.in_channels, self.hid_channels)
        self.conv2 = ConvBlock(self.hid_channels, self.hid_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2)
        self.linear1 = nn.Linear(self.hid_channels, self.hid_channels)
        self.linear2 = nn.Linear(self.hid_channels,1)


    def forward(self, input_features):
        x = input_features[-1]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

class VisibilityNet2(nn.Module):
    def __init__(self, num_ch_enc):
        super(VisibilityNet2, self).__init__()
        self.in_channels = num_ch_enc[-1]
        self.hid_channels = self.in_channels*2
        self.conv1 = ConvBlock(self.in_channels, self.hid_channels)
        #self.conv2 = ConvBlock(self.hid_channels, self.hid_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2)
        #self.linear1 = nn.Linear(self.hid_channels, self.hid_channels)
        self.linear2 = nn.Linear(self.hid_channels,1)


    def forward(self, input_features):
        x = input_features[-1]
        x = self.conv1(x)
        #x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        #x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        return x
