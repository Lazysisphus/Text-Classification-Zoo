'''
Author: Zhang Xiaozhu
Date: 2021-05-18 14:25:18
LastEditTime: 2021-05-18 16:15:21
LastEditors: Please set LastEditors
Description: 将原始的数据分词
FilePath: \text-classification-zoo\data\aclImdb\word_tokenize.py
'''

"""
原始数据为逗号分隔的csv格式，包括两列：
    text：文本
    label：标签
该脚本将对原始数据分词，并把分词后的结果作为新列添加到原始数据中
"""

import pandas as pd
from nltk import word_tokenize
from tqdm import trange

data_list = [
    "./raw_train.tsv",
    "./raw_dev.tsv",
    "./raw_test.tsv"
]

def text_tokenize(raw_data_path, new_data_path):
    """
    验证模型
    Args：
        raw_data_path：str，原始数据路径，没有分词
        new_data_path：str，分词后数据路径
    Returns：
        p|r|f|acc: float，当前模型在验证集上的p、r、f、a值
    """
    data = pd.read_csv(raw_data_path, sep="\t")
    words_list = []
    for i in trange(len(data)):
        text = str(data.text[i])
        words = word_tokenize(text)
        words_list.append(words)
    data["words"] = words_list
    data.to_csv(new_data_path, index=False)


text_tokenize(data_list[0], "./train.csv")
text_tokenize(data_list[1], "./dev.csv")
text_tokenize(data_list[2], "./test.csv")

print("Done!")