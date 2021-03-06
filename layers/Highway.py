'''
@Author: your name
@Date: 2020-06-08 19:17:35
@LastEditTime: 2020-06-08 20:05:46
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \Reddit\layers\Highway.py
'''

import torch.nn as nn
import torch.nn.functional as F
import torch


class Highway(nn.Module):
    def __init__(self, layer_num, dim=600):
        super(Highway, self).__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x