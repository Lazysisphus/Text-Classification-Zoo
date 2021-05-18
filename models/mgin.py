import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embedding import Embedding
from layers.esim_layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from layers.function_layers import get_mask, replace_masked
from layers.attention import Attention


mgin_config = {
    'embed_dim' : 300,
    'hidden_dim' : 128,
    'dropout' : 0.5,
    'highway_numlayers' : 1,
    'word_dim0' : 300,
    'word_dim1' : 300,
}


class MGIN(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(MGIN, self).__init__()
        self.embed_dim = mgin_config['embed_dim']
        self.hidden_dim = mgin_config['hidden_dim']
        self.dropout = mgin_config['dropout']
        self.num_classes = opt.polarities_dim
        self.device = opt.device

        self.embedding0 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        self.embedding1 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[1], dtype=torch.float), freeze=True)

        self.text_embedding = Embedding(mgin_config['highway_numlayers'], mgin_config['word_dim0'], mgin_config['word_dim1'])

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM, self.embed_dim, self.hidden_dim, bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_dim, self.hidden_dim), nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM, self.hidden_dim, self.hidden_dim, bidirectional=True)

        self.sen_attention = Attention(mgin_config['hidden_dim']*2, score_function='dot_product')
        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                                nn.Linear(2*6*self.hidden_dim, self.hidden_dim),
                                                nn.Tanh(),
                                                nn.Dropout(p=self.dropout),
                                                nn.Linear(self.hidden_dim, self.num_classes))
        

    def forward(self, inputs):
        max_seq_len = inputs[0].size()[1]
        '''
        ids to embedding
        '''
        body_indicies = inputs[0]
        pun_indicies = inputs[1]

        body_lengths = torch.sum(body_indicies != 0, dim=-1)
        pun_lengths = torch.sum(pun_indicies != 0, dim=-1)

        body_mask = get_mask(body_indicies, body_lengths).to(self.device) # (bsz, max_seq_len)
        pun_mask = get_mask(pun_indicies, pun_lengths).to(self.device)

        #embedded_body0 = self.embedding0(body_indicies)
        embedded_body1 = self.embedding1(body_indicies)
        
        #embedded_pun0 = self.embedding0(pun_indicies)
        embedded_pun1 = self.embedding1(pun_indicies)
        
        #embedded_body = torch.cat((embedded_body0, embedded_body1), dim=-1)
        #embedded_pun = torch.cat((embedded_pun0, embedded_pun1), dim=-1)

        # embedded_body = self.text_embedding(embedded_body0, embedded_body1)
        # embedded_pun = self.text_embedding(embedded_pun0, embedded_pun1)

        if self.dropout:
            embedded_body = self._rnn_dropout(embedded_body1)
            embedded_pun = self._rnn_dropout(embedded_pun1)

        '''
        BiLSTM
        '''
        encoded_body = self._encoding(embedded_body, body_lengths)
        encoded_pun = self._encoding(embedded_pun, pun_lengths)

        '''
        word level interaction
        '''
        word_att_body, word_att_pun = self._attention(encoded_body, body_mask, encoded_pun, pun_mask)

        enhanced_body = torch.cat([encoded_body, 
                                    word_att_body,
                                    encoded_body - word_att_body,
                                    encoded_body * word_att_body],
                                    dim=-1)
        enhanced_pun = torch.cat([encoded_pun,
                                    word_att_pun,
                                    encoded_pun - word_att_pun,
                                    encoded_pun * word_att_pun],
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

        # v_word = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        '''
        sub-sentence level interaction
        '''
        body_avg = F.avg_pool1d(projected_body.transpose(1, 2), kernel_size=projected_body.size()[1]).squeeze()
        body_max = F.max_pool1d(projected_body.transpose(1, 2), kernel_size=projected_body.size()[1]).squeeze()

        pun_avg = F.avg_pool1d(projected_pun.transpose(1, 2), kernel_size=projected_pun.size()[1]).squeeze()
        pun_max = F.max_pool1d(projected_pun.transpose(1, 2), kernel_size=projected_pun.size()[1]).squeeze()

        body_sen_rep = torch.cat([body_avg, body_max], dim=-1)
        pun_sen_rep = torch.cat([pun_avg, pun_max], dim=-1)

        body_sen_final, _ = self.sen_attention(encoded_body, pun_sen_rep)
        body_sen_final = body_sen_final.squeeze(dim=1)
        
        pun_sen_final, _ = self.sen_attention(encoded_pun, body_sen_rep)
        pun_sen_final = pun_sen_final.squeeze(dim=1)

        v = torch.cat([v_a_avg, v_a_max, body_sen_rep, v_b_avg, v_b_max, pun_sen_rep], dim=-1)
        logits = self._classification(v)
        
        return logits