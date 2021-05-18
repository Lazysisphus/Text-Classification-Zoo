'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2020-01-05 09:39:28
@LastEditors: Please set LastEditors
@LastEditTime: 2020-06-08 21:15:57
'''

from layers.attention import Attention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F


fgian_config = {
    'embed_dim' : 300,
    'hidden_dim' : 64,
    'input_drop' : 0.5,
    'num_layers' : 1
}


class AlignmentMatrix(nn.Module):
    def __init__(self, opt):
        super(AlignmentMatrix, self).__init__()
        self.opt = opt
        self.w_u = nn.Parameter(torch.Tensor(6 * fgian_config['hidden_dim'], 1))

    def forward(self, batch_size, body, pun):
        body_len = body.size(1)
        pun_len = pun.size(1)
        alignment_mat = torch.zeros(batch_size, body_len, pun_len).to(self.opt.device)
        body_chunks = body.chunk(body_len, dim=1) # a tuple - (tensor(bsz, 1, hidden_dim*2), tensor(bsz, 1, hidden_dim*2), ...)
        pun_chunks = pun.chunk(pun_len, dim=1)
        for i, body_chunk in enumerate(body_chunks):
            for j, pun_chunk in enumerate(pun_chunks):
                feat = torch.cat([body_chunk, pun_chunk, body_chunk * pun_chunk], dim=2) # (batch_size, 1, 6 * fgian_config['hidden_dim'])
                alignment_mat[:, i, j] = feat.matmul(self.w_u.expand(batch_size, -1, -1)).squeeze(-1).squeeze(-1) 
        return alignment_mat

class WIN(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(WIN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float))

        self.body_lstm = DynamicLSTM(fgian_config['embed_dim'], 
                                     fgian_config['hidden_dim'], 
                                     num_layers=1, 
                                     batch_first=True, 
                                     bidirectional=True,
                                     rnn_type='LSTM')
        self.pun_lstm = DynamicLSTM(fgian_config['embed_dim'], 
                                    fgian_config['hidden_dim'], 
                                    num_layers=1, 
                                    batch_first=True, 
                                    bidirectional=True,
                                    rnn_type='LSTM'
                                    )
        
        self.w_p2b = nn.Parameter(torch.Tensor(2 * fgian_config['hidden_dim'], 2 * fgian_config['hidden_dim']))
        self.w_b2p = nn.Parameter(torch.Tensor(2 * fgian_config['hidden_dim'], 2 * fgian_config['hidden_dim']))
        self.alignment = AlignmentMatrix(opt)
        self.dense = nn.Linear(4 * fgian_config['hidden_dim'], opt.polarities_dim)

    def forward(self, inputs):
        '''
        ids to embedding
        '''
        body_indices = inputs[0]
        pun_indices = inputs[1]
        batch_size = body_indices.size(0)
        
        body_len = torch.sum(body_indices != 0, dim=1)
        pun_len = torch.sum(pun_indices != 0, dim=1)
        
        body_embedding = self.embed(body_indices) # batch_size x seq_len x embed_dim
        pun_embedding = self.embed(pun_indices) # batch_size x seq_len x embed_dim

        '''
        coarse-grained interactive attention
        '''
        body_M, (_, _) = self.body_lstm(body_embedding, body_len)
        # body_pool = torch.sum(body_M, dim=1)
        # body_pool = torch.div(body_pool, body_len.float().unsqueeze(-1)).unsqueeze(-1) # (batch_size, 2 * fgian_config['hidden_dim'], 1)

        pun_M, (_, _) = self.pun_lstm(pun_embedding, pun_len)
        # pun_pool = torch.sum(pun_M, dim=1)
        # pun_pool = torch.div(pun_pool, pun_len.float().unsqueeze(-1)).unsqueeze(-1) # (batch_size, 2 * fgian_config['hidden_dim'], 1)

        # c_pun2body_alpha = F.softmax(body_M.matmul(self.w_p2b.expand(batch_size, -1, -1)).matmul(pun_pool), dim=1)
        # c_pun2body = torch.matmul(body_M.transpose(1, 2), c_pun2body_alpha).squeeze(-1)
        # c_body2pun_alpha = F.softmax(pun_M.matmul(self.w_b2p.expand(batch_size, -1, -1)).matmul(body_pool), dim=1)
        # c_body2pun = torch.matmul(pun_M.transpose(1, 2), c_body2pun_alpha).squeeze(-1)

        '''
        fine-grained interactive attention
        '''
        alignment_mat = self.alignment(batch_size, body_M, pun_M)
        f_pun2body = torch.matmul(body_M.transpose(1, 2), F.softmax(alignment_mat.max(2, keepdim=True)[0], dim=1)).squeeze(-1)
        f_body2pun = torch.matmul(F.softmax(alignment_mat.max(1, keepdim=True)[0], dim=2), pun_M).transpose(1, 2).squeeze(-1)

        # feat = torch.cat([c_pun2body, f_pun2body, c_body2pun, f_body2pun], dim=1)
        feat = torch.cat([f_pun2body, f_body2pun], dim=1)
        logits = self.dense(feat)

        return logits
