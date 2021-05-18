'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-18 10:21:11
@LastEditors: Please set LastEditors
@LastEditTime: 2020-06-08 21:08:32
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import Attention
from layers.dynamic_rnn import DynamicLSTM


sian_config = {
    'embed_dim' : 300,
    'hidden_dim' : 64,
    'input_drop' : 0.5,
    'num_layers' : 1
}


class SIN(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(SIN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        # self.input_drop = nn.Dropout(sian_config['input_drop'])
        self.bilstm = DynamicLSTM(sian_config['embed_dim'],
                                    sian_config['hidden_dim'],
                                    num_layers=sian_config['num_layers'],
                                    batch_first=True,
                                    bidirectional=True,
                                    rnn_type='LSTM'
                                    )

        # self.bilstm_body = DynamicLSTM(sian_config['embed_dim'], 
        #                                sian_config['hidden_dim'], 
        #                                num_layers=sian_config['num_layers'], 
        #                                batch_first=True, 
        #                                bidirectional=True
        #                                )
        # self.bilstm_pun = DynamicLSTM(sian_config['embed_dim'], 
        #                               sian_config['hidden_dim'], 
        #                               num_layers=sian_config['num_layers'], 
        #                               batch_first=True, 
        #                               bidirectional=True
        #                               )

        # self.weight1 = nn.Parameter(torch.Tensor(sian_config['hidden_dim'] * 2, sian_config['hidden_dim'] * 2))
        # self.weight2 = nn.Parameter(torch.Tensor(sian_config['hidden_dim'] * 2, 1))

        # nn.init.uniform_(self.weight1, -0.1, 0.1)
        # nn.init.uniform_(self.weight2, -0.1, 0.1)

        self.attention =Attention(sian_config['hidden_dim'] * 2, score_function='dot_product')
        # self.attention_body = Attention(sian_config['hidden_dim'] * 2, score_function='dot_product')
        # self.attention_pun = Attention(sian_config['hidden_dim'] * 2, score_function='dot_product')
        
        self.dense = nn.Linear(sian_config['hidden_dim'] * 4, opt.polarities_dim)

    def forward(self, inputs):
        '''
        ids to emb
        '''
        # full_indicies = inputs[0]
        body_indicies = inputs[1]
        pun_indicies = inputs[2]
        
        # full_emb = self.embed(full_indicies)
        body_emb = self.embed(body_indicies)
        pun_emb = self.embed(pun_indicies)

        # full_emb = self.input_drop(full_emb)
        # body_emb = self.input_drop(body_emb)
        # pun_emb = self.input_drop(pun_emb)

        '''
        bilstm
        '''
        # full_len = torch.sum(full_indicies != 0, dim=-1)
        body_len = torch.sum(body_indicies != 0, dim=-1)
        pun_len = torch.sum(pun_indicies != 0, dim=-1)
        
        # full_M, (full_ht, _) = self.bilstm(full_emb, full_len)
        body_M, (body_ht, _) = self.bilstm(body_emb, body_len)
        pun_M, (pun_ht, _) = self.bilstm(pun_emb, pun_len)

        
        '''
        attention
        '''
        # score = torch.tanh(torch.matmul(sen_M, self.weight1))
        # attention_weights = F.softmax(torch.matmul(score, self.weight2), dim=1) # attention_weights - (bsz, seq_len, 1)

        # sen_out = sen_M * attention_weights
        # sen_rep = torch.sum(sen_out, dim=1)

        # body_score = torch.tanh(torch.matmul(body_M, self.weight1))
        # pun_score = torch.tanh(torch.matmul(pun_M, self.weight1))

        # body_attention_weights = F.softmax(torch.matmul(body_score, self.weight2), dim=1)
        # pun_attention_weights= F.softmax(torch.matmul(pun_score, self.weight2), dim=1)

        # body_out = body_M * body_attention_weights
        # pun_out = pun_M * pun_attention_weights

        # body_att = torch.sum(body_out, dim=1)
        # pun_att = torch.sum(pun_out, dim=1)

        '''
        interactive attention
        '''
        body_len = torch.tensor(body_len, dtype=torch.float).to(self.opt.device)
        body_pool = torch.sum(body_M, dim=1)
        body_pool = torch.div(body_pool, body_len.view(body_len.size(0), 1))
        
        pun_len = torch.tensor(pun_len, dtype=torch.float).to(self.opt.device)
        pun_pool = torch.sum(pun_M, dim=1)
        pun_pool = torch.div(pun_pool, pun_len.view(pun_len.size(0), 1))

        body_final, _ = self.attention(body_M, pun_pool)
        body_final = body_final.squeeze(dim=1)
        
        pun_final, _ = self.attention(pun_M, body_pool)
        pun_final = pun_final.squeeze(dim=1)

        '''
        classification
        '''
        
        sen_rep = torch.cat((body_final, pun_final), dim=-1)
        logits = self.dense(sen_rep)
        
        return logits
    