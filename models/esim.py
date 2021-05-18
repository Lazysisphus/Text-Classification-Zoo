'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-12-18 16:58:39
@LastEditors: Please set LastEditors
@LastEditTime: 2020-06-08 11:35:45
'''


import torch
import torch.nn as nn

from layers.esim_layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from layers.function_layers import get_mask, replace_masked


hsin_config = {
    'embed_dim' : 300,
    'hidden_dim' : 128,
    'dropout' : 0.5,
}


class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for Natural Language Inference" by Chen et al.
    """
    def __init__(self, embedding_matrix_list, opt):
        super(ESIM, self).__init__()
        self.embed_dim = hsin_config['embed_dim']
        self.hidden_dim = hsin_config['hidden_dim']
        self.dropout = hsin_config['dropout']
        self.num_classes = opt.polarities_dim
        self.device = opt.device

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        # self.embedding1 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[1], dtype=torch.float), freeze=True)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM, self.embed_dim, self.hidden_dim, bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_dim, self.hidden_dim), nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM, self.hidden_dim, self.hidden_dim, bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                                nn.Linear(2*4*self.hidden_dim, self.hidden_dim),
                                                nn.Tanh(),
                                                nn.Dropout(p=self.dropout),
                                                nn.Linear(self.hidden_dim, self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self, inputs):
        '''
        ids to embedding
        '''
        body_indicies = inputs[0]
        pun_indicies = inputs[1]

        body_lengths = torch.sum(body_indicies != 0, dim=-1)
        pun_lengths = torch.sum(pun_indicies != 0, dim=-1)

        body_mask = get_mask(body_indicies, body_lengths).to(self.device) # (bsz, max_seq_len)
        pun_mask = get_mask(pun_indicies, pun_lengths).to(self.device)

        embedded_body = self.embedding(body_indicies)
        # embedded_body1 = self.embedding1(body_indicies)
        
        embedded_pun = self.embedding(pun_indicies)
        # embedded_pun1 = self.embedding1(pun_indicies)
        
        # embedded_body = torch.cat((embedded_body0, embedded_body1), dim=-1)
        # embedded_pun = torch.cat((embedded_pun0, embedded_pun1), dim=-1)

        if self.dropout:
            embedded_body = self._rnn_dropout(embedded_body)
            embedded_pun = self._rnn_dropout(embedded_pun)

        '''
        BiLSTM
        '''
        encoded_body = self._encoding(embedded_body, body_lengths)
        encoded_pun = self._encoding(embedded_pun, pun_lengths)

        attended_body, attended_pun = self._attention(encoded_body, body_mask, encoded_pun, pun_mask)

        enhanced_body = torch.cat([encoded_body, 
                                    attended_body,
                                    encoded_body - attended_body,
                                    encoded_body * attended_body],
                                    dim=-1)
        enhanced_pun = torch.cat([encoded_pun,
                                    attended_pun,
                                    encoded_pun - attended_pun,
                                    encoded_pun * attended_pun],
                                    dim=-1)

        projected_body = self._projection(enhanced_body)
        projected_pun = self._projection(enhanced_pun)

        if self.dropout:
            projected_body = self._rnn_dropout(projected_body)
            projected_pun = self._rnn_dropout(projected_pun)

        v_ai = self._composition(projected_body, body_lengths)
        v_bj = self._composition(projected_pun, pun_lengths)

        v_a_avg = torch.sum(v_ai * body_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(body_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * pun_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(pun_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, body_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, pun_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        
        return logits


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
