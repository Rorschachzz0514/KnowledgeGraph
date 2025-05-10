import numpy as np
from transformers import  BertModel, BertTokenizer
#from Transformer import CustomTransformer
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.optim.lr_scheduler import StepLR
from DataReader import Reader

def BertEncoder(text):
    # 加载预训练的分词器

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 加载预训练的BERT模型
    model = BertModel.from_pretrained('bert-base-uncased')

    # 使用分词器对文本进行处理
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # 获取BERT模型的输出
    outputs = model(**inputs)
    return outputs.last_hidd


dataset=Reader("./wiki/single_data_label_description.pkl",True)
print("kkk")

