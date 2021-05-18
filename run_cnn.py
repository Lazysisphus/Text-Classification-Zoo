'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2021-05-18 14:13:06
LastEditors: Please set LastEditors
LastEditTime: 2021-05-18 16:51:48
'''


import os
import math
import random
import argparse

import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import build_embedding_matrix
from data_utils import build_tokenizer, MyDataset


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


class TextCnnModel(nn.Module):
    """
    CNN模型类
    """
    def __init__(self, opt, embedding_matrix_list):
        super(TextCnnModel, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        self.input_drop = nn.Dropout(opt.dropout)
        self.convs = nn.ModuleList([nn.Conv1d(opt.emb_dim, opt.kernel_num, K) for K in [int(x) for x in opt.kernel_sizes.split(" ")]])
        self.dropout =nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.kernel_size * opt.kernel_num, opt.polarities_dim)
                
    def forward(self, inputs):
        '''
        ids to emb
        '''
        sen_indicies = inputs
        sen_feature = self.embed(sen_indicies)
        sen_feature = self.input_drop(sen_feature)

        '''
        produce feature maps
        '''
        conv_list = []
        for conv in self.convs:
            conv_L = conv(sen_feature.transpose(1, 2))
            conv_L = self.dropout(conv_L)
            conv_L = F.max_pool1d(conv_L, conv_L.size(2)).squeeze(2)
            conv_list.append(conv_L)

        sen_out = [i.view(i.size(0), -1) for i in conv_list]
        sen_out = torch.cat(sen_out, dim=1)

        '''
        classification
        '''
        logits = self.dense(sen_out)
        
        return logits
        

class TextCNN:
    """
    CNN训练类
    """
    def __init__(self):
        """
        类初始化，设置各种参数
        """
        parser = argparse.ArgumentParser()
        # 数据路径
        parser.add_argument("--data_path", default="./data/MR/", type=str)
        parser.add_argument("--train_data_path", default="./data/MR/train.csv", type=str)
        parser.add_argument("--dev_data_path", default="./data/MR/dev.csv", type=str)
        parser.add_argument("--test_data_path", default="./data/MR/test.csv", type=str)
        parser.add_argument("--max_seq_len", default=20, type=int)
        parser.add_argument("--polarities_dim", default=2, type=int)
        # 模型
        parser.add_argument("--model_name", default="TextCNN", type=str)
        # 训练
        parser.add_argument("--num_epoch", default=16, type=int)
        parser.add_argument("--batch_size", default=512, type=int)
        parser.add_argument("--log_step", default=64, type=int)
        # 优化器
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--l2reg", default=0.00001, type=float)
        parser.add_argument("--initializer", default="xavier_uniform_", type=str)
        parser.add_argument("--optimizer", default="adam", type=str)
        parser.add_argument("--adamw_epsilon", default=1e-8, type=float)
        parser.add_argument("--weight_decay", default=0.01, type=float)
        # 设备
        parser.add_argument("--device", default=None, type=str)
        parser.add_argument("--n_gpu", default=1, type=int)
        # 随机数种子
        parser.add_argument("--seed", default=42, type=int)
        # 模型超参数
        parser.add_argument("--emb_dim", default=300, type=int)
        parser.add_argument("--kernel_size", default=3, type=int)
        parser.add_argument("--kernel_sizes", default="2 3 4", type=str)
        parser.add_argument("--kernel_num", default=256, type=int)
        parser.add_argument("--mlp_dim", default=128, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        # 预训练词向量
        parser.add_argument(
            "--wv_paths", 
            default="../language_models/glove.840B.300d.txt",
            type=str, 
            help="预训练词向量路径（可多个，逗号间隔），如果为None表示不使用预训练词向量"
            )
        self.opt = parser.parse_args()
        if self.opt.wv_paths is not None:
            self.opt.wv_dims = "300" # 预训练词向量维度（可多个，逗号间隔）
            self.opt.wv_types = "glove" # 预训练词向量类型（可多个，逗号间隔），有word2vec、glove、fasttext三种供选择
            self.opt.wv_names = "glove840B" # 预训练词向量名字（可多个，逗号间隔）
            # self.opt.wv_binaries = "False,False" # 预训练词向量保存格式是否是二进制（可多个，空格间隔）

        initializers = {
            "xavier_uniform_": torch.nn.init.xavier_uniform_,
            "xavier_normal_": torch.nn.init.xavier_normal,
            "orthogonal_": torch.nn.init.orthogonal_
        }
        optimizers = {
            "adadelta": torch.optim.Adadelta, # default lr=1.0
            "adagrad": torch.optim.Adagrad, # default lr=0.01
            "adam": torch.optim.Adam, # default lr=0.001
            "adamax": torch.optim.Adamax, # default lr=0.002
            "asgd": torch.optim.ASGD, # default lr=0.01
            "rmsprop": torch.optim.RMSprop, # default lr=0.01
            "sgd": torch.optim.SGD
        }
        self.opt.initializer = initializers[self.opt.initializer]
        self.opt.optimizer = optimizers[self.opt.optimizer]

        if self.opt.device is None:
            self.opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.opt.device = torch.device(self.opt.device)
        
        def _set_seed(num):
            """
            设置随机数种子
            Args：
                num：int，随机数种子值
            Returns：
                None
            """
            os.environ["PYTHONHASHSEED"] = str(num)
            np.random.seed(num)
            random.seed(num)
            torch.manual_seed(num)
            torch.cuda.manual_seed(num)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        _set_seed(self.opt.seed)

        # tokenizer
        self.tokenizer = build_tokenizer(
            max_seq_len=self.opt.max_seq_len,
            tokenizer_dat=self.opt.data_path + "tokenizer.dat",
            data_paths=[self.opt.train_data_path, self.opt.dev_data_path, self.opt.test_data_path],
            mode="word"
            )

        # 预训练词向量
        if self.opt.wv_paths is None:
            embedding_matrix_list = None
        else:
            embedding_matrix_list = []
            wv_paths = [str(x).strip() for x in self.opt.wv_paths.split(",")]
            wv_dims = [int(x) for x in self.opt.wv_dims.split(",")]
            wv_types = [str(x).strip() for x in self.opt.wv_types.split(",")]
            wv_names = [str(x).strip() for x in self.opt.wv_names.split(",")]
            # wv_binaries = self.opt.wv_binaries.split(" ")
            assert len(wv_paths) == len(wv_dims) == len(wv_types) == len(wv_names), "预训练词向量参数设置错误!"
            for i in range(len(wv_paths)):
                tmp_embedding_matrix = build_embedding_matrix(
                    word2idx=self.tokenizer.word2idx,
                    wv_path=wv_paths[i],
                    wv_dim=wv_dims[i],
                    wv_type=wv_types[i],
                    wv_initial_matrix="{0}_{1}_matirx.dat".format(str(wv_dims[i]), wv_names[i])
                )
                embedding_matrix_list.append(tmp_embedding_matrix)
            self.opt.vocab_num = len(self.tokenizer.word2idx) + 2

        # model
        self.model = TextCnnModel(self.opt, embedding_matrix_list).to(self.opt.device)

        # data
        self.trainset = MyDataset(self.opt.train_data_path, self.tokenizer)
        self.devset = MyDataset(self.opt.dev_data_path, self.tokenizer)
        self.testset = MyDataset(self.opt.test_data_path, self.tokenizer)
            
        if self.opt.device.type == "cuda":
            print("cuda memory allocated: {:.4f} MB". \
                format(torch.cuda.memory_allocated(device=self.opt.device.index) / 1024 / 1024))
        self._print_args()

    def _print_args(self):
        """
        打印各种参数
        Args：
            None
        Retuens：
            None
        """
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print("n_trainable_params: {0}, n_nontrainable_params: {1}". \
            format(n_trainable_params, n_nontrainable_params))
        print("> training arguments:")
        for arg in vars(self.opt):
            print(">>> {0}: {1}".format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        """
        初始化模型可训练参数
        Args：
            None
        Returns：
            None
        """
        for child in self.model.children():
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        self.opt.initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def fit(self):
        """
        训练函数
        Args：
            None
        Returns：
            None
        """
        self._reset_params()
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dataset=self.devset, batch_size=self.opt.batch_size, shuffle=False)

        # multi gpu
        if self.opt.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer_grouped_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(
            optimizer_grouped_parameters, 
            lr=self.opt.learning_rate, 
            weight_decay=self.opt.l2reg
            )
        
        # metrics to select model
        max_val_acc, max_val_f1 = 0, 0

        path = None
        self.model.zero_grad()
        for epoch in range(self.opt.num_epoch):
            print(">" * 66)
            print("epoch: {}".format(epoch))
            global_step = 0
            n_correct, n_total, loss_total = 0, 0, 0
            
            for _, sample_batched in enumerate(train_data_loader):
                self.model.train()
                global_step += 1
                
                inputs = torch.as_tensor(sample_batched["x_data"], dtype=torch.long).to(self.opt.device)
                targets = torch.as_tensor(sample_batched['y_label'], dtype=torch.long).to(self.opt.device)
                outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)
                # loss.requires_grad = True
                if self.opt.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    print("train_loss: {:.4f}, train_acc: {:.4f}".format(train_loss, train_acc))
                    
                    # validate per log_steps
                    val_precision, val_recall, val_f1, val_acc = self._evaluate(dev_data_loader)
                    print("> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f}, val_acc: {:.4f}". \
                        format(val_precision, val_recall, val_f1, val_acc))
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        if not os.path.exists("./state_dict/"):
                            os.mkdir("./state_dict/")
                        path = "./state_dict/{0}_{1}_val_metric_{2}". \
                            format(self.opt.model_name, str(self.opt.seed), round(val_acc, 4))
                        torch.save(self.model.state_dict(), path)
                        print(">> saved: {}".format(path))
        return path

    def _evaluate(self, data_loader):
        """
        验证模型
        Args：
            data_loader：需验证数据集的迭代器
        Returns：
            p|r|f|acc: float，当前模型在验证集上的p、r、f、a值
        """
        n_correct, n_total = 0, 0
        targets_all, outputs_all = None, None
        self.model.eval()
        with torch.no_grad():
            for _, sample_batched in enumerate(data_loader):
                inputs = torch.as_tensor(sample_batched["x_data"], dtype=torch.long).to(self.opt.device)
                targets = torch.as_tensor(sample_batched['y_label'], dtype=torch.long).to(self.opt.device)
                outputs = self.model(inputs)
                
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)

                if targets_all is None:
                    targets_all = targets
                    outputs_all = outputs
                else:
                    targets_all = torch.cat((targets_all, targets), dim=0)
                    outputs_all = torch.cat((outputs_all, outputs), dim=0)
        
        precision = metrics.precision_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu())
        recall = metrics.recall_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu())
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu())
        acc = n_correct / n_total
        return precision, recall, f1, acc
    
    def predict(self, best_model_path, label_flag=True):
        if label_flag is True:
            test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
            self.model.load_state_dict(torch.load(best_model_path))
            self.model.eval()
            test_precision, test_recall, test_f1, test_acc = self._evaluate(test_data_loader)
            print(">> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}". \
                format(test_precision, test_recall, test_f1, test_acc))


if __name__ == "__main__":
    obj = TextCNN()
    best_model_path = obj.fit()
    obj.predict(best_model_path, True)