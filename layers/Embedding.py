'''
@Author: your name
@Date: 2020-06-08 19:13:21
@LastEditTime: 2020-06-08 19:22:43
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \Reddit\layers\Embedding.py
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

from .Highway import Highway

class Embedding(nn.Module):
    def __init__(self, highway_layers, word_dim0, word_dim1):
        super(Embedding, self).__init__()
        self.highway = Highway(highway_layers, word_dim0 + word_dim1)

    def forward(self, word_emb0, word_emb1):
        emb = torch.cat([word_emb0, word_emb1], dim=2)
        emb = self.highway(emb)

        return emb
