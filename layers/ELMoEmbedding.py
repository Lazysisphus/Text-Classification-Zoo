'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-12-03 18:15:50
@LastEditors: Zhang Xiaozhu
@LastEditTime: 2019-12-05 15:28:40
'''


import torch.nn as nn
from allennlp.modules.elmo import Elmo


class ElmoLayer(nn.Module):
    def __init__(self, max_len, num_output_representations=1, dropout=0, requires_grad=False):
        super(ElmoLayer, self).__init__()
        self.max_len = max_len
        
        options_file = './layers/ELMo_model/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
        weight_file = './layers/ELMo_model/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
        self.embedding = Elmo(options_file, weight_file, num_output_representations=num_output_representations, dropout=dropout, requires_grad=requires_grad)

    # input : (bsz, max_seq_len, character_dim==50)
    def forward(self, character_ids_of_sens):
        elmo_outputs = self.embedding(character_ids_of_sens)
        elmo_embeddings = elmo_outputs['elmo_representations'] # (bsz, max_seq_len, 1024)
        
        return elmo_embeddings
    