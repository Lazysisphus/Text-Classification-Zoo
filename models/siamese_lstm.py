'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-18 10:21:11
@LastEditors  : Zhang Xiaozhu
@LastEditTime : 2020-01-06 11:32:37
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dynamic_rnn import DynamicLSTM

siamese_lstm_config = {
    'embed_dim' : 300,
    'hidden_dim' : 64,
    'input_drop' : 0.5,
    'num_layers' : 1
}


class SIAMESE_LSTM(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(SIAMESE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        # self.input_drop = nn.Dropout(siamese_lstm_config['input_drop'])

        self.lstm = DynamicLSTM(siamese_lstm_config['embed_dim'],
                                siamese_lstm_config['hidden_dim'],
                                num_layers=siamese_lstm_config['num_layers'],
                                batch_first=True,
                                bidirectional=False
                                )
        # self.w = nn.Parameter(torch.Tensor(siamese_lstm_config['hidden_dim'] * 2))
        # self.norm = nn.BatchNorm1d(siamese_lstm_config['hidden_dim'] * 4)
        # self.tanh = nn.Tanh()
        # self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(siamese_lstm_config['hidden_dim'] * 2, opt.polarities_dim)
        

    def forward(self, inputs):
        '''
        ids to emb
        '''
        sen1_indicies = inputs[0]
        sen2_indicies = inputs[1]

        sen1_emb = self.embed(sen1_indicies)
        sen2_emb = self.embed(sen2_indicies)

        sen1_emb = self.input_drop(sen1_emb0)
        sen2_emb = self.input_drop(sen2_emb0)

        '''
        lstm
        '''
        sen1_len = torch.sum(sen1_indicies != 0, dim=-1)
        sen2_len = torch.sum(sen2_indicies != 0, dim=-1)

        _, (sen1_ht, _) = self.lstm(sen1_emb, sen1_len)
        _, (sen2_ht, _) = self.lstm(sen2_emb, sen2_len)

        '''
        classification
        '''
        # sen_pair = torch.cat((sen1_out, sen2_out), dim=-1)
        # sen_pair = torch.cat((sen1_ht[0], sen2_ht[0], torch.mul(sen1_ht[0], sen2_ht[0]), torch.abs(sen1_ht[0] - sen2_ht[0])), dim=-1)
        sen_pair = torch.cat((sen1_ht[0], sen2_ht[0]), dim=-1)
        logits = self.dense(sen_pair)
        
        return logits
        