'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-18 10:21:11
@LastEditors  : Zhang Xiaozhu
@LastEditTime : 2020-01-06 15:50:40
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dynamic_rnn import DynamicLSTM


siamese_bert_config = {
    'embed_dim' : 300,
    'hidden_dim' : 64,
    'input_drop' : 0.5,
    'num_layers' : 1
}


class SIAMESE_BILSTM_ATTENTION(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(SIAMESE_BILSTM_ATTENTION, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        # self.input_drop = nn.Dropout(siamese_bert_config['input_drop'])
        self.bilstm_body = DynamicLSTM(siamese_bert_config['embed_dim'],
                                       siamese_bert_config['hidden_dim'],
                                       num_layers=siamese_bert_config['num_layers'],
                                       batch_first=True,
                                       bidirectional=True
                                       )
        self.weight1 = nn.Parameter(torch.Tensor(siamese_bert_config['hidden_dim'] * 2, siamese_bert_config['hidden_dim'] * 2))
        self.weight2 = nn.Parameter(torch.Tensor(siamese_bert_config['hidden_dim'] * 2, 1))

        nn.init.uniform_(self.weight1, -0.1, 0.1)
        nn.init.uniform_(self.weight2, -0.1, 0.1)
                      
        self.dense = nn.Linear(siamese_bert_config['hidden_dim'] * 4, opt.polarities_dim)

    def forward(self, inputs):
        '''
        ids to emb
        '''
        body_indicies = inputs[0]
        pun_indicies = inputs[1]
        body_emb = self.embed(body_indicies)
        pun_emb = self.embed(pun_indicies)
        # body_emb = self.input_drop(body_emb)
        # pun_emb = self.input_drop(pun_emb)

        '''
        bilstm
        '''
        body_len = torch.sum(body_indicies != 0, dim=-1)
        pun_len = torch.sum(pun_indicies != 0, dim=-1)
        
        body_M, (body_ht, _) = self.bilstm(body_emb, body_len) # sen_M - (bsz, seq_len, hidden_dim * 2)
        pun_M, (pun_ht, _) = self.bilstm(pun_emb, pun_len) # sen_M - (bsz, seq_len, hidden_dim * 2)

        '''
        attention
        '''
        body_score = torch.tanh(torch.matmul(body_M, self.weight1))
        pun_score = torch.tanh(torch.matmul(pun_M, self.weight1))

        body_attention_weights = F.softmax(torch.matmul(body_score, self.weight2), dim=1)
        pun_attention_weights= F.softmax(torch.matmul(pun_score, self.weight2), dim=1)

        body_out = body_M * body_attention_weights
        pun_out = pun_M * pun_attention_weights

        body_rep = torch.sum(body_out, dim=1)
        pun_rep = torch.sum(pun_out, dim=1)

        '''
        classification
        '''
        sen_rep = torch.cat((body_rep, pun_rep), dim=-1)
        logits = self.dense(sen_rep)
        