'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-16 17:09:58
@LastEditors: Please set LastEditors
@LastEditTime: 2020-06-08 19:36:09
'''

'''
CNN
'''
from models.cnn import CNN
from models.gcnn import GCNN
from models.dpcnn import DPCNN
from models.rcnn import RCNN
from models.rnn_cnn import RNN_CNN

'''
RNN + ATTENTION
'''
from models.lstm import LSTM
from models.bilstm import BILSTM
from models.bilstm_attention import BILSTM_ATTENTION
from models.bilstm_ssa import BILSTM_SSA

'''
SIAMESE MODELS
'''
from models.siamese_cnn import SIAMESE_CNN
from models.siamese_lstm import SIAMESE_LSTM
from models.siamese_bilstm_attention import SIAMESE_BILSTM_ATTENTION

'''
INTERACTIVE MODELS
'''
from models.sin import SIN
from models.win import WIN
from models.esim import ESIM
from models.mgin import MGIN

'''
PRE-TRAINED MODELS
'''
from models.bert_spc import BERT_SPC