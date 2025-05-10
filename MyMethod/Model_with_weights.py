from Bert_Transformer import BertTransformer
from DataReader import Reader
from Mapper import Mapper
import torch
import math
import random
from create_batch import get_pair_batch_test, toarray, get_pair_batch_train_common
from transformers import  BertModel, BertTokenizer
from model import BiLSTM_Attention

from NodeClassifier import NodeClassifier
import numpy as np
# import args
#from dataset import Reader
# import utils
from create_batch import get_pair_batch_test, toarray, get_pair_batch_train_common,get_pair_all_test_common
import torch
#from model import BiLSTM_Attention
import torch.nn as nn
import os
import logging
import math
# import time
import argparse
import random
from transformers import  BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




class ModelWithWeights(nn.Module):
    def __init__(self, transformer_small, transformer_combine, classifier, CAGED_model,mapper,args):
        super(ModelWithWeights, self).__init__()
        self.transformer_small = transformer_small
        self.transformer_combine = transformer_combine
        self.classifier = classifier
        self.CAGED_model = CAGED_model
        self.mapper=mapper
        # 定义可训练的参数w_1和w_2
        self.w_1 = nn.Parameter(torch.tensor(1.0))
        self.w_2 = nn.Parameter(torch.tensor(1.0))
        self.args=args
    def forward(self, batch_h, batch_r, batch_t, bert_embedding,batch_size):
        transformer_cls = []
        for embedding in bert_embedding:
            head_embedding = embedding[0][0]
            head_mask = embedding[0][1]
            relation_embedding = embedding[1][0]
            relation_mask = embedding[1][1]
            tail_embedding = embedding[2][0]
            tail_mask = embedding[2][1]
            # combined_embeddings=torch.cat([head_embedding,relation_embedding,tail_embedding],dim=-1)
            # 首先分别根据掩码计算每一个bert的embedding
            head_output = self.transformer_small(head_embedding, src_key_padding_mask=~head_mask.bool())
            relation_output = self.transformer_small(relation_embedding, src_key_padding_mask=~relation_mask.bool())
            tail_output = self.transformer_small(tail_embedding, src_key_padding_mask=~tail_mask.bool())
            combined_embeddings = torch.cat([head_output, relation_output, tail_output], dim=-1)
            combined_mask = torch.logical_or(torch.logical_or(head_mask, relation_mask), tail_mask)
            # 讲所有的embedding通过拼接的方式汇总，再过一遍transformer
            transformer_x = self.transformer_combine(combined_embeddings, src_key_padding_mask=~combined_mask.bool())
            transformer_single_cls = transformer_x[:, 0, :][0]
            transformer_cls.append(transformer_single_cls)

        transformer_cls = torch.stack(transformer_cls)
        mapper_trans=self.mapper(transformer_cls)
        out, out_att = self.CAGED_model(batch_h, batch_r, batch_t)
        #out_att = out_att.reshape(out_att.shape[0], -1, 2 * 3 * self.args.BiLSTM_hidden_size)
        out_att = out_att.reshape(batch_size, -1, 2 * 3 * self.args.BiLSTM_hidden_size)
        pos_z0 = out_att[:, 0, :]
        pos_z1 = out_att[:, 1, :]

        # 使用可训练的w_1和w_2
        outputs = self.classifier(mapper_trans + self.w_1 * pos_z0 + self.w_2 * pos_z1)
        return outputs,out,out_att

# # 在训练函数中创建模型实例
# def train(args, dataset, device):
#     # 加载预训练模型
#     transformer_small = torch.load('./model/transformer_small.pth')
#     transformer_combine = torch.load('./model/transformer_combine.pth')
#     classifier = torch.load('./model/classifier.pth')
#     CAGED_model = torch.load("./model/CAGED.pth")
#
#     # 创建包含可训练w_1和w_2的模型
#     model_with_weights = ModelWithWeights(transformer_small, transformer_combine, classifier, CAGED_model)
#
#     # 定义优化器，包含w_1和w_2
#     optimizer = torch.optim.Adam(model_with_weights.parameters(), lr=0.0001)
#
#     # 其他代码保持不变
#     # ...
#
#     # 在训练循环中使用model_with_weights而不是单独的模型
#     for k in range(args.max_epoch):
#         for it in range(num_iterations):
#             # 获取批次数据
#             batch_h, batch_r, batch_t, batch_size, bert_embedding, label = get_pair_batch_train_common(args, dataset, it, train_idx, args.batch_size, args.num_neighbor)
#             label = torch.tensor(label)
#
#             # 前向传播
#             outputs = model_with_weights(batch_h, batch_r, batch_t, bert_embedding)
#
#             # 计算损失
#             loss = criterion1(outputs, label)
#
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             print("Loss:", loss.item())
#
# # 在测试函数中使用model_with_weights
# def test(args, dataset, train_idx):
#     print("test")
#     with torch.no_grad():  # 关闭梯度计算
#         # 加载预训练模型
#         transformer_small = torch.load('./model/transformer_small.pth')
#         transformer_combine = torch.load('./model/transformer_combine.pth')
#         classifier = torch.load('./model/classifier.pth')
#         CAGED_model = torch.load("./model/CAGED.pth")
#
#         # 创建包含可训练w_1和w_2的模型
#         model_with_weights = ModelWithWeights(transformer_small, transformer_combine, classifier, CAGED_model)
#
#         # 加载数据
#         batch_h, batch_r, batch_t, bert_embedding, label = get_pair_all_test_common(args, dataset, train_idx, args.batch_size, args.num_neighbor)
#         label = torch.tensor(label)
#
#         # 前向传播
#         outputs = model_with_weights(batch_h, batch_r, batch_t, bert_embedding)
#
#         _, preds = torch.max(outputs, 1)  # 获取预测结果
#         accuracy = accuracy_score(label, preds)
#         precision = precision_score(label, preds, average='weighted')
#         recall = recall_score(label, preds, average='weighted')
#         f1 = f1_score(label, preds, average='weighted')
#         print(f"Accuracy: {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"F1 Score: {f1:.4f}")
#
# if __name__ == '__main__':
#     main()