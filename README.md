# Text-Classification-Zoo
a text classification zoo.

## 数据集
为方便对比模型性能，采用情感分析领域的经典数据集：
1. Imdb：电影评论英文数据集，训练集和测试集各包括2.5万条数据，二分类，正负例平衡。从训练集中随机抽取了5000条数据构建验证集。数据集文本长度稍长。
2. SST-2：来自songyingxin/TextClassification，电影评论英文数据集，二分类，切分为了训练集、验证集和测试集。为短文本数据集。

## 使用步骤
1. 数据准备。提供csv格式（逗号分隔）的训练集（raw_train.csv）、验证集（raw_dev.csv）、测试集（raw_test.csv），每个数据集包括两列，“text”列为文本，“label”列为标签（从0开始，0、1、2...）。三个数据集中测试集为可选项，可以只使用run_xxx.py脚本训练模型，而不测试。若有测试集，对于含标签测试集，训练好的模型将通过训练类的predict函数输出模型在测试集的评价指标及预测分类结果；对于不含标签测试集，模型只输出分类结果。
2. 数据分词。在./data/文件夹下新建文件夹，将任务数据放入其中。在./data/文件夹下运行word_tokenize.py对原始数据分词，得到模型的输入数据train.csv、dev.csv以及test.csv。
3. 模型训练及预测。选择run_xxx.py文件，修改数据路径、预训练词向量路径等参数，训练模型并测试。

Model        |Precision    |Recall        |F1_Score     |Accuracy
-------------|-------------| -------------|-------------|------------- 
TextCNN      |92.43        |94.29         |93.35        |92.43
LSTM         |             |              |             |
