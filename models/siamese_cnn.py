'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-18 10:21:11
@LastEditors  : Zhang Xiaozhu
@LastEditTime : 2020-01-06 11:31:24
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


siamese_cnn_config = {
    'emb_dim': 300,
    'kernel_size': 3,
    'kernel_sizes': [2, 3, 5],
    'kernel_num': 64,
    'mlp_dim': 128,
    'dropout': 0.5,
    'input_drop' : 0.5
}


class SIAMESE_CNN(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(SIAMESE_CNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        # self.input_drop = nn.Dropout(siamese_cnn_config['input_drop'])
        self.convs = nn.ModuleList([nn.Conv1d(siamese_cnn_config['emb_dim'], siamese_cnn_config['kernel_num'], K) for K in siamese_cnn_config['kernel_sizes']])
        self.dropout =nn.Dropout(siamese_cnn_config['dropout'])

        self.dense = nn.Linear(siamese_cnn_config['kernel_size'] * siamese_cnn_config['kernel_num'] * 2, opt.polarities_dim)
                
    def forward(self, inputs):
        '''
        ids to emb
        '''
        sen1_indicies = inputs[0]
        sen2_indicies = inputs[1]

        sen1_feature = self.embed(sen1_indicies)
        sen2_feature = self.embed(sen2_indicies)

        # sen1_feature = self.input_drop(sen1_emb)
        # sen2_feature = self.input_drop(sen2_emb)

        '''
        produce feature maps
        '''
        conv1_list = []
        for conv in self.convs:
            conv_L = conv(sen1_feature.transpose(1, 2))
            conv_L = self.dropout(conv_L)
            conv_L = F.max_pool1d(conv_L, conv_L.size(2)).squeeze(2)
            conv1_list.append(conv_L)

        sen1_out = [i.view(i.size(0), -1) for i in conv1_list]
        sen1_out = torch.cat(sen1_out, dim=1)

        conv2_list = []
        for conv in self.convs:
            conv_L = conv(sen2_feature.transpose(1, 2))
            conv_L = self.dropout(conv_L)
            conv_L = F.max_pool1d(conv_L, conv_L.size(2)).squeeze(2)
            conv2_list.append(conv_L)

        sen2_out = [i.view(i.size(0), -1) for i in conv2_list]
        sen2_out = torch.cat(sen2_out, dim=1)

        '''
        classification
        '''
        # sen_pair = torch.cat((sen1_out, sen2_out, torch.mul(sen1_out, sen2_out), torch.abs(sen1_out - sen2_out)), dim=-1)
        sen_pair = torch.cat((sen1_out, sen2_out), dim=-1)
        logits = self.dense(sen_pair)
        
        return logits
        