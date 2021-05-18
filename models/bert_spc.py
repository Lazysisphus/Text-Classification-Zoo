'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-22 20:35:01
@LastEditors  : Zhang Xiaozhu
@LastEditTime : 2020-01-03 22:24:06
'''


import torch
import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(opt.pretrained_model_dim, opt.polarities_dim)

    def forward(self, inputs):
        x_sen = inputs[2]
        att_mask = inputs[3]

        _, pooler_output = self.bert(x_sen, att_mask)

        pooler_output = self.dropout(pooler_output)
        logits = self.dense(pooler_output)

        return logits
